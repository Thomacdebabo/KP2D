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

    def __init__(self, shape, fov, r_min, r_max, device = 'cpu',amp = 50, artifact_amp = 200, patch_ratio = 0.95, scaling_amplitude = 0.1, max_angle_div = 18, super_resolution = 2,
                 preprocessing_gradient = True, add_row_noise = True, add_normal_noise = False, add_artifact = True, add_sparkle_noise = False, blur = False, add_speckle_noise = False, normalize = True):
        #super resolution helps mitigate introduced artifacts by the coordinate transforms
        self.super_resolution = super_resolution
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
        self.artifact_amp = artifact_amp
        self.amp = amp

        self.artifact_width = 2*self.super_resolution

        self.patch_ratio = patch_ratio
        self.scaling_amplitude = scaling_amplitude
        self.max_angle = pi / max_angle_div

        self.preprocessing_gradient = preprocessing_gradient
        self.add_row_noise = add_row_noise
        self.add_normal_noise = add_normal_noise
        self.add_sparkle_noise = add_sparkle_noise
        self.blur = blur
        self.add_speckle_noise = add_speckle_noise
        self.normalize = normalize
        self.post_noise = False
        self.add_artifact = add_artifact



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

    def filter(self, img, amp=30):
        filtered = img
        if self.preprocessing_gradient:
            filtered = gradient_curve(filtered)

        if self.add_row_noise:
            noise = create_row_noise_torch(torch.clip(filtered, 0, 50),amp=amp, device=self.device) * 2
            for i in range(round(self.super_resolution)):
                noise = torch.nn.functional.conv2d(noise, self.kernel, bias=None, stride=[1, 1], padding='same')
            noise = noise*self.super_resolution

            filtered = torch.clip(filtered + noise,0,255)

        if self.add_normal_noise:
            noise = torch.clip((torch.rand(filtered.shape)-0.5)*amp,0,255).to(self.device)
            for i in range(round(self.super_resolution)):
                noise = torch.nn.functional.conv2d(noise, self.kernel, bias=None, stride=[1, 1], padding='same')
            noise = noise*self.super_resolution
            filtered = torch.clip(filtered + noise,0,255)

        if self.add_artifact:
            artifact = torch.zeros(filtered.shape).to(self.device)
            attenuation = torch.linspace(self.artifact_amp/0.3*0.1,self.artifact_amp/0.3,filtered.shape[3])*(torch.rand(filtered.shape[3])+1)

            noise = torch.clip((torch.rand(filtered.shape).to(self.device) - 0.7) * attenuation[:,None].to(self.device), 0, 255)
            mid = int(self.shape[1] * self.super_resolution / 2)
            artifact[:, :, :, mid - self.artifact_width:mid + self.artifact_width] = noise[:, :, :, mid - self.artifact_width:mid + self.artifact_width]
            for i in range(round(self.super_resolution)):
                artifact = torch.nn.functional.conv2d(artifact, self.kernel, bias=None, stride=[1, 1], padding='same')
                artifact = torch.nn.functional.conv2d(artifact, self.kernel, bias=None, stride=[1, 1], padding='same')
            filtered = torch.clip(filtered + artifact,0,255)

        if self.add_sparkle_noise:
            filtered = add_sparkle(filtered, self.kernel, device=self.device)

        if self.blur:
            filtered = torch.nn.functional.conv2d(filtered, self.kernel, bias=None, stride=[1,1], padding='same')

        if self.add_speckle_noise:
            filtered = torch.clip(filtered * (0.5+0.6*create_speckle_noise(filtered, self.kernel, device=self.device)), 0, 255)

        if self.normalize:
            filtered = filtered/filtered.max()*255

        return filtered

    def add_noise_function(self, sample):
        img = sample['image']
        aug = sample['image_aug']
        noise = torch.clip((torch.rand(img[0,:,:].shape) - 0.5) * self.amp, 0, 255).unsqueeze(0).unsqueeze(0)
        noise_aug = torch.clip((torch.rand(aug[0,:,:].shape) - 0.5) * self.amp, 0, 255).unsqueeze(0).unsqueeze(0)
        for i in range(round(self.super_resolution)):
            noise = torch.nn.functional.conv2d(noise, self.kernel, bias=None, stride=[1, 1], padding='same')
            noise_aug = torch.nn.functional.conv2d(noise_aug, self.kernel, bias=None, stride=[1, 1], padding='same')

        noise = noise * self.super_resolution
        noise_aug = noise_aug * self.super_resolution

        sample['image'] = img + noise.squeeze(0)
        sample['image_aug'] = aug + noise_aug.squeeze(0)
        return sample
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
                                       patch_ratio=self.patch_ratio,
                                       scaling_amplitude=self.scaling_amplitude,
                                       max_angle=self.max_angle)
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
        amp = self.amp*(0.3+torch.rand(1))
        sample['image'] = self.filter(img, amp=amp).to(img.dtype)
        sample['image_aug'] = self.filter(img_aug, amp=amp).to(img_aug.dtype)
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
    for r in range(x.shape[2]):
        noise[:,:,r,:] =(torch.rand(x.shape[3])-0.5).to(device)/(torch.sum(x[:,:,r,:])/torch.tensor(np.random.normal(2500,1000,1)).to(device)+1)
    return noise*amp

def create_speckle_noise(x,conv_kernel, device = 'cpu'):
    noise = torch.clip(torch.rand(x.shape, device = device)-0.5,-1,1)
    speckle = torch.nn.functional.conv2d(noise, conv_kernel, bias=None, stride=[1, 1], padding='same')
    return speckle

def add_sparkle(x, conv_kernel, device = 'cpu'):
    sparkle = torch.clip((x-180)*5,0,255)
    #sparkle = torch.clip((kornia.morphology.dilation(x, kernel, iterations=2).astype('int8')-50)*2-np.random.normal(20,100,x.shape),0,255)
    sparkle = torch.nn.functional.conv2d(sparkle, conv_kernel, bias=None, stride=[1,1], padding='same')
    x = torch.clip(x*0.7+sparkle*0.3,0,255)
    return x

def gradient_curve(x,a=0.25, x0=0.5):
    b = (1 - a*x0)/(1-x0)-a
    x = x/255
    return (torch.max(x*a,torch.tensor(0)) + torch.max((x-x0),torch.tensor(0))*b)*255



