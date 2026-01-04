import math
import torch
import torch.nn as nn
import numpy as np

class GaussianBlurOperator(nn.Module):
    """
    Fourier-domain Gaussian blur with circular boundary conditions.

    Forward:   x -> ifft2( fft2(x) * H )
    Adjoint:   y -> ifft2( fft2(y) * conj(H) )
    ATA:       z -> ifft2( fft2(z) * |H|^2 )

    where H(fx, fy) = exp( -2 * pi^2 * sigma^2 * (fx^2 + fy^2) )
    sigma is in spatial pixel units (same units as the image grid spacing).
    """
    def __init__(self, x, sigma=1.0, pixel_size=1.0,eta=1e-6):
        """
        Args:
            x (torch.Tensor): a sample tensor to infer image size (expects [..., H, W]).
            sigma (float): spatial standard deviation in pixels.
            pixel_size (float): spatial sampling step (default 1.0 pixel).
        """
        super().__init__()
        assert x.ndim >= 2, "Input must have at least [..., H, W] dimensions."
        H, W = x.shape[-2], x.shape[-1]

        device = x.device
        dtype = torch.float32  # frequency response is real-valued; complex created on the fly

        # Frequency grids in cycles per pixel (compatible with torch.fft.fft2)
        fy = torch.fft.fftfreq(H, d=pixel_size, device=device)
        fx = torch.fft.fftfreq(W, d=pixel_size, device=device)
        fy, fx = torch.meshgrid(fy, fx, indexing="ij")
        radius2 = fx**2 + fy**2

        # Analytic Fourier transform of a spatial Gaussian (std = sigma):
        # H(f) = exp(-2 * pi^2 * sigma^2 * |f|^2)
        Hfreq = torch.exp(-2.0 * (math.pi ** 2) * (sigma ** 2) * radius2).to(dtype)

        # Shape to broadcast over batch/channel: [1, 1, H, W]
        Hfreq = Hfreq.unsqueeze(0).unsqueeze(0)

        # Register as buffers so they move with .to(device) / .cuda()
        self.Hfreq = Hfreq                # real, >= 0
        self.Hfreq_conj = Hfreq           # conj(H) == H for real Gaussian
        self.Hfreq_abs2 =  Hfreq * Hfreq  # |H|^2
        self.eta2 = eta**2 # noise standarddeviation

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply blur in Fourier domain: ifft2( fft2(x) * H ).
        """
        X = torch.fft.fft2(data)
        Y = X * self.Hfreq  # broadcast over batch/channel
        y = torch.fft.ifft2(Y)
        # For real-valued inputs/PSF, imaginary part should be ~0 (numerical noise).
        return y.real

    def adjoint(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply adjoint: ifft2( fft2(y) * conj(H) ).
        For a real, symmetric Gaussian, conj(H) == H, so adjoint == forward.
        """
        Y = torch.fft.fft2(data)
        Z = Y * self.Hfreq_conj
        z = torch.fft.ifft2(Z)
        return z.real

    def ATA(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply A^T A efficiently in Fourier domain: ifft2( fft2(x) * |H|^2 ).
        """
        X = torch.fft.fft2(data)
        Y = X * self.Hfreq_abs2
        y = torch.fft.ifft2(Y)
        return y.real
    


    def PreCondition(self, data: torch.Tensor, t: float = 0) -> torch.Tensor:

        """
            Preconditioner that approximates the pseudo-inverse of the operator
            Implements (A^T A/eta^2 + (1/t^2) I)^(-1) in Fourier domain.
        """
        Y = torch.fft.fft2(data)
        if t==0:
            gain = self.Hfreq_abs2/self.eta2  
        else:
            gain = self.Hfreq_abs2/self.eta2  + 1.0/(t**2)

        gain = 1/gain

        Xhat = gain * Y
        x_hat = torch.fft.ifft2(Xhat)
        return x_hat.real
    
    def NoiseModulation(self, data: torch.Tensor, t: float = 0, rtol: float = 1e-3) -> torch.Tensor:

        """
            Implements (A^T A/eta^2 + (1/t^2) I)^(-1/2) in Fourier domain.
        """
        Y = torch.fft.fft2(data)
        if t==0:
            gain = self.Hfreq_abs2/self.eta2  
        else:
            gain = self.Hfreq_abs2/self.eta2  + 1.0/(t**2)

        gain = torch.sqrt(1/gain)

        Xhat = gain * Y
        x_hat = torch.fft.ifft2(Xhat)
        return x_hat.real
    

class Inpainting(nn.Module):
    def __init__(self, mask,eta=1e-6):
        super().__init__()
        
        self.mask = mask
        self.eta2 = eta**2

    def forward(self, data):
        return self.mask*data 

    def adjoint(self, data):
        return self.mask * data
    
    def ATA(self,data):
        return  self.adjoint(self.forward(data))
    
    def PreCondition(self, x,t):
        
        bmat = 1.0/(self.mask/self.eta2 + 1.0/t**2)
        return bmat*x
    
    def NoiseModulation (self, n,t):
        
        bmat = 1.0/(self.mask/self.eta2 + 1.0/t**2)
        bmat = torch.sqrt(bmat)
        return bmat*n
    


    import math
import torch
import torch.nn as nn

class MotionBlurOperator(nn.Module):
    """
    Fourier-domain motion blur with circular boundary conditions.

    Forward:   x -> ifft2( fft2(x) * H )
    Adjoint:   y -> ifft2( fft2(y) * conj(H) )
    ATA:       z -> ifft2( fft2(z) * |H|^2 )

    The PSF is constructed in the spatial domain as a uniform line segment
    (length in pixels, angle in degrees), then transformed to frequency domain
    via fft2(fftshift(psf)).
    """
    def __init__(self,
                 x: torch.Tensor,
                 length: float = 15.0,
                 theta_deg: float = 0.0,
                 oversample: int = 8,
                 pixel_size: float = 1.0,
                 eta: float = 1e-6):
        """
        Args:
            x (torch.Tensor): sample tensor to infer spatial size; expects [..., H, W].
            length (float): motion blur length in pixels.
            theta_deg (float): motion direction in degrees (0° = +x, CCW positive).
            oversample (int): supersampling factor for anti-aliased PSF construction.
            pixel_size (float): spatial sampling step (kept for interface parity; PSF uses pixel units).
            eta (float): noise standard deviation; eta^2 is used in PreCondition/NoiseModulation.
        """
        super().__init__()
        assert x.ndim >= 2, "Input must have at least [..., H, W] dimensions."
        H, W = x.shape[-2], x.shape[-1]
        device = x.device
        dtype = torch.float32  # build PSF in float32; FFT will promote to complex

        # Build spatial PSF (centered) and convert to frequency response H
        psf = self._motion_psf_linear_torch(H, W, length, theta_deg,
                                            oversample=oversample,
                                            device=device, dtype=dtype)

        # Normalize to unit sum (photometric consistency)
        ssum = psf.sum()
        if ssum <= 0:
            # Fallback to delta if degenerate
            psf = torch.zeros_like(psf)
            psf[H // 2, W // 2] = 1.0
        else:
            psf = psf / ssum

        # Compute OTF: shift to center then FFT
        Hfreq = torch.fft.fft2(torch.fft.fftshift(psf))  # shape [H, W], complex

        # Shape for broadcasting over [B, C, H, W]
        Hfreq = Hfreq.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        # Register buffers (so they move with .to(device) and are saved)
        self.register_buffer("Hfreq", Hfreq)                          # complex
        self.register_buffer("Hfreq_conj", torch.conj(Hfreq))         # complex
        self.register_buffer("Hfreq_abs2", (torch.abs(Hfreq) ** 2))   # real

        # noise variance
        self.eta2 = float(eta) ** 2
        # Store params (optional, for reference)
        self.length = float(length)
        self.theta_deg = float(theta_deg)
        self.oversample = int(max(1, oversample))
        self.pixel_size = float(pixel_size)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply motion blur in Fourier domain: ifft2( fft2(x) * H ).
        """
        X = torch.fft.fft2(data)
        Y = X * self.Hfreq
        y = torch.fft.ifft2(Y)
        return y.real

    def adjoint(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply adjoint: ifft2( fft2(y) * conj(H) ).
        """
        Y = torch.fft.fft2(data)
        Z = Y * self.Hfreq_conj
        z = torch.fft.ifft2(Z)
        return z.real

    def ATA(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply A^T A: ifft2( fft2(x) * |H|^2 ).
        """
        X = torch.fft.fft2(data)
        Y = X * self.Hfreq_abs2
        y = torch.fft.ifft2(Y)
        return y.real

    def PreCondition(self, data: torch.Tensor, t: float = 0.0, eps: float = 1e-12) -> torch.Tensor:
        """
        Implements (A^T A / eta^2 + (1/t^2) I)^(-1) in the Fourier domain.
        """
        Y = torch.fft.fft2(data)
        if t == 0:
            gain = self.Hfreq_abs2 / self.eta2
        else:
            gain = self.Hfreq_abs2 / self.eta2 + (1.0 / (t ** 2))
        gain = 1.0 / (gain + eps)  # invert diagonal operator
        Xhat = gain * Y            # real gain * complex spectrum
        x_hat = torch.fft.ifft2(Xhat)
        return x_hat.real

    def NoiseModulation(self, data: torch.Tensor, t: float = 0.0, eps: float = 1e-12) -> torch.Tensor:
        """
        Implements (A^T A / eta^2 + (1/t^2) I)^(-1/2) in the Fourier domain.
        """
        Y = torch.fft.fft2(data)
        if t == 0:
            gain = self.Hfreq_abs2 / self.eta2
        else:
            gain = self.Hfreq_abs2 / self.eta2 + (1.0 / (t ** 2))
        gain = torch.sqrt(1.0 / (gain + eps))
        Xhat = gain * Y
        x_hat = torch.fft.ifft2(Xhat)
        return x_hat.real

    @staticmethod
    @torch.no_grad()
    def _motion_psf_linear_torch(H: int,
                                 W: int,
                                 length: float,
                                 theta_deg: float,
                                 oversample: int = 8,
                                 device: str = "cpu",
                                 dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Build a centered 2D uniform linear motion PSF using supersampling and
        bilinear splatting for anti-aliased lines.

        Args:
            H, W: output PSF size
            length: motion length in pixels
            theta_deg: direction (degrees), 0° = +x, CCW positive
            oversample: supersampling factor (>=1)
            device, dtype: tensor placement and type

        Returns:
            psf tensor of shape [H, W], dtype=float, sum~=1 (not normalized here).
        """
        s = int(max(1, oversample))
        Hs, Ws = H * s, W * s

        # High-res accumulation buffer (flat for scatter efficiency)
        psf_hi = torch.zeros((Hs * Ws,), device=device, dtype=dtype)

        # Line endpoints in the high-res grid (centered)
        cy, cx = (Hs - 1) / 2.0, (Ws - 1) / 2.0
        theta = torch.tensor(theta_deg, device=device, dtype=torch.float64) * (torch.pi / 180.0)
        dx, dy = torch.cos(theta), torch.sin(theta)
        Lh = (float(length) * s) / 2.0

        x0, y0 = cx - Lh * dx, cy - Lh * dy
        x1, y1 = cx + Lh * dx, cy + Lh * dy

        # Temporal samples along the segment (uniform "box" shutter)
        T = max(1, int(max(64, 4 * Lh)))  # more samples for longer blur
        t = torch.linspace(0.0, 1.0, T, device=device, dtype=torch.float64)

        xs = x0 + t * (x1 - x0)
        ys = y0 + t * (y1 - y0)
        w = torch.ones_like(t, dtype=torch.float64) / T  # uniform shutter

        # Bilinear splatting to four neighbors per sample
        ix0 = torch.floor(xs).to(torch.long)
        iy0 = torch.floor(ys).to(torch.long)
        fx = (xs - ix0.to(xs.dtype)).to(dtype)  # [0,1) frac
        fy = (ys - iy0.to(ys.dtype)).to(dtype)

        ix1 = ix0 + 1
        iy1 = iy0 + 1

        # Masks to keep indices inside bounds
        def in_bounds(ix, iy):
            return (ix >= 0) & (ix < Ws) & (iy >= 0) & (iy < Hs)

        # neighbor coordinates and weights
        neigh = [
            (ix0, iy0, (1 - fx) * (1 - fy)),
            (ix1, iy0, (    fx) * (1 - fy)),
            (ix0, iy1, (1 - fx) * (    fy)),
            (ix1, iy1, (    fx) * (    fy)),
        ]

        w = w.to(dtype)

        for ix, iy, wt in neigh:
            m = in_bounds(ix, iy)
            if m.any():
                idx = (iy[m] * Ws + ix[m]).to(torch.long)
                vals = (wt[m] * w[m]).to(dtype)
                psf_hi.scatter_add_(0, idx, vals)

        # Reshape to 2D and downsample by averaging sxs blocks
        psf_hi = psf_hi.view(Hs, Ws)
        if s > 1:
            psf = psf_hi.view(H, s, W, s).mean(dim=(1, 3))
        else:
            psf = psf_hi

        return psf
    
class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, margin=(16, 16)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(img,
                              mask_shape=(mask_h, mask_w),
                              image_size=self.image_size,
                              margin=self.margin)
        return mask, t, tl, w, wh

    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(1, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask, t, th, w, wl = self._retrieve_box(img)
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask
        
def random_sq_bbox(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t+h, l:l+w] = 0

    return mask, t, t+h, l, l+w


        