import numpy as np
from kp2d.datasets.augmentations import sample_homography, warp_homography
from math import pi
import torch
from kp2d.utils.image import image_grid


#f and a are parameters to make picture not clipping
def pol_2_cart(source, fov, r_min, r_max, epsilon=1e-14, f= 1, a = 0):

    effective_range = r_max - r_min
    ang = source[:,:, 0] * fov / 2 * torch.pi / 180
    r = (source[:,:, 1] + 1 + a)*effective_range + r_min*f

    temp = torch.polar(r, ang)

    source[:,:, 1] = ((temp.real-r_min*f)/effective_range/f - 1)
    source[:,:, 0] = (temp.imag)/effective_range/f
    return source

def cart_2_pol(source, fov, r_min, r_max, epsilon=0, f= 1, a = 0):
    effective_range = r_max-r_min
    x = source[:,:, 0].clone()*effective_range * f
    y = ((source[:,:, 1].clone() + 1)*effective_range + r_min) * f

    source[:,:, 1] = ((torch.sqrt(x * x + y * y + epsilon) - r_min*f)/effective_range - 1 -a)
    source[:,:, 0] = (torch.arctan(x / (y + epsilon)) / torch.pi * 2 / fov * 180)
    return source


def to_torch(img, device = 'cpu'):
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)

def to_numpy(img):
    return (img.permute(0,2,3,1).squeeze(0).cpu().numpy()).astype(np.uint8)

class NoiseUtility():

    def __init__(self, shape, fov, r_min, r_max, device = 'cpu'):
        #super resolution helps mitigate introduced artifacts by the coordinate transforms
        self.super_resolution = 2
        self.r_min = r_min
        self.r_max = r_max
        self.shape = shape
        self.fov = fov
        self.device = device
        self.map, self.map_inv = self.init_map()
        self.kernel = self.init_kernel()
        H, W = self.shape
        self.x_cart_scale = W/2
        self.y_cart_scale = H/2
        self.amp = 20



    def init_kernel(self):
        kernel = torch.tensor(
            [[0, 0, 0, 0, 0], [0, 0.25, 0.5, 0.25, 0], [0.5, 1, 1, 1, 0.5], [0, 0.25, 0.5, 0.25, 0], [0, 0, 0, 0, 0]])
        kernel = (kernel / torch.sum(kernel)).unsqueeze(0).unsqueeze(0).to(self.device)
        return kernel

    #TODO: use same function as in the train script
    def init_map(self):
        H, W = self.shape
        source_grid = image_grid(1, H*self.super_resolution, W*self.super_resolution,
                                 dtype = torch.float32,
                                 device=self.device,
                                 ones=False, normalized=True).clone().permute(0, 2, 3, 1)
        source_grid_2 = image_grid(1, H, W,
                                 dtype = torch.float32,
                                 device=self.device,
                                 ones=False, normalized=True).clone().permute(0, 2, 3, 1)

        map = pol_2_cart(source_grid_2.clone().squeeze(0), self.fov, r_min=self.r_min, r_max=self.r_max).unsqueeze(0)
        map_inv = cart_2_pol(source_grid.clone().squeeze(0), self.fov,r_min=self.r_min, r_max=self.r_max).unsqueeze(0)
        return map, map_inv

    def filter(self, img):
        filtered = gradient_curve(img)
        noise = create_row_noise_torch(torch.clip(filtered, 2, 50),amp=self.amp, device=self.device) * 2

        filtered = filtered + noise
        #filtered = add_sparkle(filtered, self.kernel, device=self.device)
        filtered = torch.nn.functional.conv2d(filtered, self.kernel, bias=None, stride=[1,1], padding='same')

        #filtered = add_sparkle(filtered, self.kernel, device=self.device)
        #filtered = (filtered * 0.75 + img * 0.25)

        filtered = torch.clip(filtered * (0.7+0.5*create_speckle_noise(filtered, self.kernel, device=self.device)), 0, 255)
        return filtered

    def sim_2_real_filter(self, img):
        img = to_torch(img, device=self.device)
        if img.shape.__len__() == 5:
            mapped = self.pol_2_cart_torch(img.permute(0,4,2,3,1).squeeze(-1))[:,0,:,:].unsqueeze(0)
        else:
            mapped = self.pol_2_cart_torch(img.unsqueeze(-1))

        mapped = self.filter(mapped)
        mapped = self.cart_2_pol_torch(mapped).squeeze(0)
        if img.shape.__len__() == 5:
            return torch.stack((mapped, mapped, mapped), axis=1).to(img.dtype)
        else:
            return mapped.to(img.dtype)

    # functions dedicated to working with the samples coming from the dataloader
    def pol_2_cart_sample(self, sample):
        img = to_torch(np.array(sample['image'])[:,:,0], device= self.device)
        mapped = self.pol_2_cart_torch(img)
        sample['image'] = mapped.to(img.dtype)
        return sample

    def augment_sample(self, sample):
        orig_type = sample['image'].dtype
        img = sample['image']
        _,_,H, W = img.shape

        homography = sample_homography([H, W], perspective=False, scaling = True,
                                       patch_ratio=0.75,
                                       scaling_amplitude=0.5,
                                       max_angle=pi/2)
        homography = torch.from_numpy(homography).float().to(self.device)
        source_grid = image_grid(1, H, W,
                                 dtype=img.dtype,
                                 device=self.device,
                                 ones=False, normalized=True).clone().permute(0, 2, 3, 1)

        source_warped = warp_homography(source_grid, homography)
        source_img = torch.nn.functional.grid_sample(img, source_warped, align_corners=True)

        sample['image'] = img.to(orig_type)
        sample['image_aug'] = source_img.to(orig_type)
        sample['homography'] = homography
        return sample

    def filter_sample(self, sample):
        img = sample['image']
        img_aug = sample['image_aug']

        sample['image'] = self.filter(img).to(img.dtype)
        sample['image_aug'] = self.filter(img_aug).to(img_aug.dtype)
        return sample

    def cart_2_pol_sample(self, sample):
        img = sample['image']
        img_aug = sample['image_aug']
        mapped = self.cart_2_pol_torch(img).squeeze(0).squeeze(0)
        mapped_aug = self.cart_2_pol_torch(img_aug).squeeze(0).squeeze(0)
        sample['image'] = torch.stack((mapped, mapped, mapped), axis=0).to(img.dtype)
        sample['image_aug'] = torch.stack((mapped_aug, mapped_aug, mapped_aug), axis=0).to(img.dtype)

        #cv2.imshow("img", mapped.to(device).numpy()  / 255)
        #cv2.imshow("img aug", mapped_aug.to(device).numpy()  / 255)
        #cv2.waitKey(0)
        return sample

    # torch implementations of cartesian/polar conversions
    def pol_2_cart_torch(self, img):
        return torch.nn.functional.grid_sample(img, self.map_inv, mode='bilinear', padding_mode='zeros',
                                                align_corners=True)

    def cart_2_pol_torch(self, img):
        return torch.nn.functional.grid_sample(img, self.map, mode='bilinear', padding_mode='zeros',
                                                align_corners=True)

def create_row_noise_torch(x, amp= 50, device='cpu'):
    noise = x.clone().to(device)
    amp = torch.randn(1)*20+amp
    for r in range(x.shape[2]):
        noise[:,:,r,:] =(torch.randn(x.shape[3])*amp-amp/2 + torch.randn(x.shape[3])*amp/2+amp/2).to(device)/(torch.sum(x[:,:,r,:])/torch.tensor(np.random.normal(2500,1000,1)).to(device)+1)
    return noise

def create_speckle_noise(x,conv_kernel, device = 'cpu'):
    noise = torch.clip(torch.rand(x.shape, device = device)-0.5,-1,1)
    speckle = torch.nn.functional.conv2d(noise, conv_kernel, bias=None, stride=[1, 1], padding='same')
    return noise

def add_sparkle(x, conv_kernel, device = 'cpu'):
    sparkle = torch.clip((x-100)-torch.randn(x.shape).to(device),0,255)
    #sparkle = torch.clip((kornia.morphology.dilation(x, kernel, iterations=2).astype('int8')-50)*2-np.random.normal(20,100,x.shape),0,255)
    sparkle = torch.nn.functional.conv2d(sparkle, conv_kernel, bias=None, stride=[1,1], padding='same')
    x = torch.clip(x+sparkle,0,255)
    return x

def gradient_curve(x,a=0.4, x0=0.5):
    b = 1 + x0 - a
    x = x/255
    return (torch.max(x*a,torch.tensor(0)) + torch.max((x-0.5)*b,torch.tensor(0)))*255



