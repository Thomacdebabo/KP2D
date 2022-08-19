import numpy as np
from kp2d.datasets.augmentations import sample_homography, warp_homography
from math import pi
import torch
from kp2d.utils.image import image_grid



def pol_2_cart(source, fov, r_min, r_max, epsilon=1e-14, f= 1, a = 0):
    """
    Transform coordinate grid from polar coordinates to cartesian coordinates

    :param source: 2D coordinate grid in polar coordinates
    :param fov: field of view of sonar device
    :param r_min: minimum perceived range of sonar device
    :param r_max: maximum perceived range of sonar device
    :param epsilon: small value to make numerical calculations stable
    :param f: scaling factor for image
    :param a: offset value for image
    :return: 2D coordinate grid in cartesian coordinates
    """
    # f and a are parameters to make picture not clipping
    effective_range = r_max - r_min
    ang = source[:,:, 0] * fov / 2 * torch.pi / 180
    r = (source[:,:, 1] + 1 + a)*effective_range + r_min*f

    temp = torch.polar(r, ang)

    source[:,:, 1] = ((temp.real-r_min*f)/effective_range/f - 1)
    source[:,:, 0] = (temp.imag)/effective_range/f
    return source

def cart_2_pol(source, fov, r_min, r_max, epsilon=0, f= 1, a = 0):
    """
    Transform coordinate grid from cartesian coordinates to polar coordinates

    :param source: 2D coordinate grid in cartesian coordinates
    :param fov: field of view of sonar device
    :param r_min: minimum perceived range of sonar device
    :param r_max: maximum perceived range of sonar device
    :param epsilon: small value to make numerical calculations stable
    :param f: scaling factor for image
    :param a: offset value for image
    :return: 2D coordinate grid in polar coordinates
    """
    effective_range = r_max-r_min
    x = source[:,:, 0].clone()*effective_range * f
    y = ((source[:,:, 1].clone() + 1)*effective_range + r_min) * f

    source[:,:, 1] = ((torch.sqrt(x * x + y * y + epsilon) - r_min*f)/effective_range - 1 -a)
    source[:,:, 0] = (torch.arctan(x / (y + epsilon)) / torch.pi * 2 / fov * 180)
    return source


def to_torch(img, device = 'cpu'):
    """
    Quick function to convert greyscale image read by either PIL or CV2 to be used in torch framework.
    WARNING: be careful PIL and CV2 have use different orders of dimensions

    :param img: numpy array [H,W]
    :param device: computing device (either cpu or cuda)
    :return: torch array [1,1,H,W] on specified device
    """
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)

def to_numpy(img):
    """
    Quick function to convert RGB torch array to numpy array in PIL format (to use in CV2 use permute).
    WARNING: be careful PIL and CV2 have use different orders of dimensions
    :param img: torch array [1,1,H,W] on specified device
    :return: numpy array [3,H,W]
    """
    return (img.permute(0,2,3,1).squeeze(0).cpu().numpy()).astype(np.uint8)

class NoiseUtility():
    """This utility is designed to """
    def __init__(self, shape, fov, r_min, r_max, device = 'cpu',amp = 50, artifact_amp = 200, artifact_width= 2, patch_ratio = 0.95, scaling_amplitude = 0.1, max_angle_div = 18, super_resolution = 2,
                 preprocessing_gradient = True, add_row_noise = True, add_normal_noise = False, add_artifact = True, add_sparkle_noise = False, blur = False, add_speckle_noise = False, normalize = True):


        #super resolution helps mitigate introduced artifacts by the coordinate transforms
        self.super_resolution = super_resolution

        # physical sonar parameters
        self.r_min = r_min
        self.r_max = r_max
        self.fov = fov
        self.shape = shape

        # run inits
        self.device = device
        self.map, self.map_inv = self.init_map()
        self.kernel = self.init_kernel()

        # noise parameters
        self.artifact_amp = artifact_amp
        self.artifact_width = artifact_width * self.super_resolution # adjust for increased resolution in the calculations
        self.amp = amp

        # augmentation parameters
        self.patch_ratio = patch_ratio
        self.scaling_amplitude = scaling_amplitude
        self.max_angle = pi / max_angle_div

        # paremeters to turn on/off any part of the noise simulation (all bool)
        self.preprocessing_gradient = preprocessing_gradient

        self.add_row_noise = add_row_noise
        self.add_normal_noise = add_normal_noise
        self.add_sparkle_noise = add_sparkle_noise
        self.add_speckle_noise = add_speckle_noise
        self.add_artifact = add_artifact

        self.blur = blur
        self.normalize = normalize

        # add noise after both transforms for ablation study
        self.post_noise = False

    def init_kernel(self):
        """Creates Kernel which creates an horizontal smearing effect in a convolution"""
        kernel = torch.tensor(
            [[0, 0, 0, 0, 0], [0, 0.25, 0.5, 0.25, 0], [0.5, 1, 1, 1, 0.5], [0, 0.25, 0.5, 0.25, 0], [0, 0, 0, 0, 0]])
        kernel = (kernel / torch.sum(kernel)).unsqueeze(0).unsqueeze(0).to(self.device)
        return kernel

    def init_map(self):
        """Creates mapping grids for quick mapping with pytorchs grid_sample
        IMPORTANT: the maps work inversely to the transformations.
        This means:
         To make a map to convert from polar to cartesian coordinates we have to put a grid through
         the cartesian to polar transformation. This is due to the way the grid_sample function works in pytorch.

         So if you need to debug these functions (I hope not since I should have fixed it) you will have to be aware
         which function is used in which transformation."""
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

    def filter(self, img, amp=30, artifact_amp = 200):
        """
        Applies sonar noise to simulated images. The class offers a lot of different styles of noises and artifacts
        which can be applied if needed. Can be turned on or off in the class instantiation.

        :param img: image torch_array[1,1,H,W] which we want to apply the noise to
        :param amp: amplitude of noise to be applied int [0-255]
        :param artifact_amp: amplitude of artifact int [0-255]
        :return: image with applied noise
        """
        filtered = img

        if self.preprocessing_gradient:
            filtered = gradient_curve(filtered)

        if self.add_row_noise:
            noise = create_row_noise_torch(torch.clip(filtered, 0, 50),amp=amp, device=self.device) * 2
            for i in range(round(self.super_resolution)):
                noise = torch.nn.functional.conv2d(noise, self.kernel, bias=None, stride=[1, 1], padding='same')
            noise = noise*self.super_resolution
            filtered = torch.clip(filtered + noise,0,255)

        #TODO: evaluate normal against row noise to see if it is actually doing anything
        # (has to be done once there is a real ealuation set)
        if self.add_normal_noise:
            noise = torch.clip((torch.rand(filtered.shape)-0.5)*amp,0,255).to(self.device)
            for i in range(round(self.super_resolution)):
                noise = torch.nn.functional.conv2d(noise, self.kernel, bias=None, stride=[1, 1], padding='same')
            noise = noise*self.super_resolution
            filtered = torch.clip(filtered + noise,0,255)

        if self.add_artifact:
            artifact = create_artifact(filtered.shape, self.device, artifact_amp, self.artifact_width)
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
        """
        Adds noise to sample (used after transformations as a comparison to row noise)
        :param sample: dict with both image and image_aug
        :return: sample with added noise
        """
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
        """
        Function that applies sonar noise to an image given as numpy array

            Used in inference or to debug.
        :param img: input image [H,W] numpy array
        :return: output image with noise as [1,1,H,W] torch array
        """
        img = to_torch(img, device=self.device)
        if img.shape.__len__() == 5:
            mapped = self.pol_2_cart_torch(img.permute(0,4,2,3,1).squeeze(-1))[:,0,:,:].unsqueeze(0)
        else:
            mapped = self.pol_2_cart_torch(img.unsqueeze(-1))

        mapped = self.filter(mapped, amp=self.amp, artifact_amp=self.artifact_amp)
        mapped = self.cart_2_pol_torch(mapped).squeeze(0)
        if img.shape.__len__() == 5:
            return torch.stack((mapped, mapped, mapped), axis=1).to(img.dtype)
        else:
            return mapped.to(img.dtype)

    # functions dedicated to working with the samples coming from the dataloader

    def augment_sample(self, sample):
        """
        Apply homography to input sample and save original, augmented and homography in same sample.

        :param sample: dict containing an image to be augmented
        :return: dict containing original image, homography and augmented image
        """
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
        """
        Adds noise to sample. Both augmented and normal image get the same style of noise but independent of each other.

        :param sample: dict containing original image, homography and augmented image
        :return: dict with same keys as input
        """
        img = sample['image']
        img_aug = sample['image_aug']

        # adds some randomness to amplitude of noise
        amp = self.amp*(0.3+torch.rand(1)).item()
        artifact_amp = self.artifact_amp*torch.rand(1).item()

        sample['image'] = self.filter(img, amp=amp,artifact_amp=artifact_amp).to(img.dtype)
        sample['image_aug'] = self.filter(img_aug, amp=amp, artifact_amp=artifact_amp).to(img_aug.dtype)
        return sample

    def pol_2_cart_sample(self, sample):
        """
        Transforms sample from polar to cartesian coordinates

        :param sample: dict containing an image to be transformed to cartesian coordinates
        :return:
        """
        img = to_torch(np.array(sample['image'])[:,:,0], device= self.device)
        mapped = self.pol_2_cart_torch(img)
        sample['image'] = mapped.to(img.dtype)
        return sample

    def cart_2_pol_sample(self, sample):
        """
        Transforms sample from cartesian to polar coordinates

        :param sample: dict containing an image and it's augmented counterpart to be transformed to polar coordinates
        :return:
        """
        img = sample['image']
        img_aug = sample['image_aug']

        mapped = self.cart_2_pol_torch(img).squeeze(0).squeeze(0)
        mapped_aug = self.cart_2_pol_torch(img_aug).squeeze(0).squeeze(0)

        sample['image'] = torch.stack((mapped, mapped, mapped), axis=0).to(img.dtype)
        sample['image_aug'] = torch.stack((mapped_aug, mapped_aug, mapped_aug), axis=0).to(img.dtype)

        return sample

    # torch implementations of cartesian/polar conversions

    def pol_2_cart_torch(self, img):
        return torch.nn.functional.grid_sample(img, self.map_inv, mode='bilinear', padding_mode='zeros',
                                                align_corners=True)

    def cart_2_pol_torch(self, img):
        return torch.nn.functional.grid_sample(img, self.map, mode='bilinear', padding_mode='zeros',
                                                align_corners=True)

# Static noise adding functinos
#TODO: check if we should apply rowwise attenuation in polar or cartesian coordinates
def create_row_noise_torch(x, amp= 50, device='cpu'):
    amp = torch.tensor(amp)
    noise = x.clone().to(device)
    for r in range(x.shape[2]):
        noise[:,:,r,:] =(torch.rand(x.shape[3])-0.5).to(device)/(torch.sum(x[:,:,r,:])/torch.tensor(np.random.normal(2500,1000,1)).to(device)+1)
    return noise*amp.to(device)

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

def create_artifact(shape, device, artifact_amp, artifact_width):
    artifact = torch.zeros(shape).to(device)
    attenuation = torch.linspace(artifact_amp  * torch.rand(1).item()*0.5, artifact_amp, shape[3]) * (
                torch.rand(shape[3]))

    noise = torch.clip((torch.ones(shape).to(device)) * attenuation[:, None].to(device), 0,
                       artifact_amp)
    mid = int(shape[3] / 2)

    artifact[:, :, :, mid -artifact_width:mid + artifact_width] = noise[:, :, :,mid - artifact_width:mid + artifact_width]
    return artifact


