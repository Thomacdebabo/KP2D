from kp2dsonar.models.KeypointNetwithIOLoss import KeypointNetwithIOLoss
from kp2dsonar.datasets.noise_model import pol_2_cart, cart_2_pol
import torch

class KeypointNetWithIOLossSonar(KeypointNetwithIOLoss):
    def __init__(self,noise_util, **kwargs):
        self.noise_util = noise_util
        super().__init__(**kwargs)

    def _warp_homography_batch(self, sources, homographies):
        """Batch warp keypoints given homographies.

        Parameters
        ----------
        sources: torch.Tensor (B,H,W,C)
            Keypoints vector.
        homographies: torch.Tensor (B,3,3)
            Homographies.

        Returns
        -------
        warped_sources: torch.Tensor (B,H,W,C)
            Warped keypoints vector.
        """
        B, H, W, _ = sources.shape
        warped_sources = []

        for b in range(B):
            source = sources[b].clone()
            source = source.view(-1, 2)

            source = pol_2_cart(source.unsqueeze(0),
                                self.noise_util.fov,
                                self.noise_util.r_min,
                                self.noise_util.r_max).squeeze(0)

            source = torch.addmm(homographies[b, :, 2], source, homographies[b, :, :2].t())
            source.mul_(1 / source[:, 2].unsqueeze(1))

            source = cart_2_pol(source.unsqueeze(0),
                                self.noise_util.fov,
                                self.noise_util.r_min,
                                self.noise_util.r_max).squeeze(0)

            source = source[:, :2].contiguous().view(H, W, 2)

            warped_sources.append(source)

        return torch.stack(warped_sources, dim=0)

