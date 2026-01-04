import numpy as np
import torch.nn as nn
import torch

#@title Define A and adjoint of A operators
class cg_block(nn.Module):
    def __init__(self, cgIter, cgTol):
        super(cg_block, self).__init__()
        self.cgIter = cgIter
        self.cgTol = cgTol
        
    def forward(self, lhs, rhs, x0):
        fn=lambda a,b: torch.abs(torch.sum(torch.conj(a)*b,axis=[-1,-2,-3]))
        x = x0
        r = rhs-lhs(x0)
        p = r
        rTr = fn(r,r)
        eps=torch.tensor(1e-10)
        for i in range(self.cgIter):
            Ap = lhs(p)
            alpha=rTr/(fn(p,Ap)+eps)
            x = x +  alpha[:,None,None,None] * p
            r = r -  alpha[:,None,None,None] * Ap
            rTrNew = fn(r,r)
            if torch.sum(torch.sqrt(rTrNew+eps)) < self.cgTol:
                
                break
                
            beta = rTrNew / (rTr+eps)
            p = r + beta[:,None,None,None] * p
            rTr=rTrNew
           
        return x



class sense_v1(nn.Module):
    def __init__(self, cgIter):
        super().__init__()
        
        self.cgIter = cgIter
        self.cg = cg_block(self.cgIter, 1e-3)#1e-12
        

    def forward(self, img, csm, mask):
        cimg = img*csm
        cimg = torch.fft.fftshift(cimg,dim=[-1,-2])
        mcksp = torch.fft.fft2(cimg,dim=[-1,-2],norm="ortho")
        mcksp = torch.fft.ifftshift(mcksp,dim=[-1,-2])
        usksp = mcksp * mask
        return usksp
        
    def adjoint(self, ksp, csm):
        ksp = torch.fft.fftshift(ksp,dim=[-1,-2])
        img = torch.fft.ifft2(ksp,dim=[-1,-2],norm="ortho")
        img = torch.fft.ifftshift(img,dim=[-1,-2])
        cs_weighted_img = torch.sum(img*torch.conj(csm),1,True)
        return cs_weighted_img

    
    def ATA(self, img, csm, mask):
        cimg = img*csm
        cimg = torch.fft.fftshift(cimg,dim=[-1,-2])
        mcksp = torch.fft.fft2(cimg,dim=[-1,-2],norm="ortho")
        mcksp = torch.fft.ifftshift(mcksp,dim=[-1,-2])
        usksp = mcksp * mask
        usksp = torch.fft.fftshift(usksp,dim=[-1,-2])
        usimg = torch.fft.ifft2(usksp,dim=[-1,-2],norm="ortho")
        usimg = torch.fft.ifftshift(usimg,dim=[-1,-2])
        cs_weighted_img = torch.sum(usimg*torch.conj(csm),1,True)
        return cs_weighted_img
    
    
        
    def NoiseModulation(self, n, mask,score_std,inference_std):
        
        n = torch.fft.fftshift(n,dim=[-1,-2])
        N = torch.fft.fft2(n,dim=[-1,-2],norm="ortho")
        N = torch.fft.ifftshift(N,dim=[-1,-2])
        if score_std ==0:
             gain = 1.0/torch.sqrt(mask/inference_std**2 )
        else:
            gain = 1.0/torch.sqrt(mask/inference_std**2 + 1/(score_std**2))

        N = N * gain
        
        nWeigthed = torch.fft.fftshift(N,dim=[-1,-2])
        nWeigthed = torch.fft.ifft2(nWeigthed,dim=[-1,-2],norm="ortho")
        nWeigthed = torch.fft.ifftshift(nWeigthed,dim=[-1,-2])
        return nWeigthed
    
    
    def init(self, x0, rhs, inference_std, score_std, csm, mask):
        
        lhs = lambda x: (1/(inference_std**2))*self.ATA(x, csm, mask) + (1/(score_std**2))*x
        out = self.cg(lhs, rhs, x0)
        
        return out
    
    def ls(self, x0, rhs, csm, mask,score_std,inference_std):
        
        lhs = lambda x: (1/(inference_std**2))*self.ATA(x, csm, mask) + (1/(score_std**2))*x
        out = self.cg(lhs, rhs, x0)
        
        return out
    