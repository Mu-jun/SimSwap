import cv2
import sys
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
import os
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# detransformer = transforms.Compose([
#         transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
#         transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
#     ])


if __name__ == '__main__':
    opt = TestOptions().parse()

    start_epoch, epoch_iter = 1, 0
    crop_size = opt.crop_size

    torch.nn.Module.dump_patches = True
    
    model = create_model(opt)
    model.eval()

    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.2, det_size=(640,640))
    with torch.no_grad():
        pic_a = opt.pic_a_path
        img_a = Image.open(pic_a).convert('RGB')
#         img_a_whole = cv2.imread(pic_a)
#         if img_a_whole is None:
#             import sys
#             sys.exit("pic_a is None")
#         tmp_img = app.get(img_a_whole,crop_size)
#         if tmp_img is None:
#             import sys
#             sys.exit("pic_a face is None")
#         img_a_align_crop, _ = tmp_img
#         img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
        img_a = transformer_Arcface(img_a)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        pic_b = opt.pic_b_path
        img_b_whole = cv2.imread(pic_b)
        if img_b_whole is None:
            sys.exit("pic_b is None")
        tmp_img = app.get(img_b_whole,crop_size)
        if tmp_img is None:
            sys.exit("pic_b face is None")
        img_b_align_crop, b_mat = tmp_img
#         sys.exit(f'{len(img_b_align_crop)}')
        img_b_align_crop_pil = Image.fromarray(cv2.cvtColor(img_b_align_crop[0],cv2.COLOR_BGR2RGB)) 
        img_b = transformer(img_b_align_crop_pil)
        img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()
        img_att = img_att.cuda()

        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)

        ############## Forward Pass ######################
        swap_result = model(None, img_att, latend_id, None, True)[0]
        
        norm = SpecificNorm()
        pasring_model = None
        swaped_img, mat, source_img = swap_result, b_mat[0], img_att
        oriimg = img_b_whole
        
        target_image_list = []
        img_mask_list = []
        
        swaped_img = swaped_img.cpu().detach().numpy().transpose((1, 2, 0))
        img_white = np.full((crop_size,crop_size), 255, dtype=float)

        # inverse the Affine transformation matrix
        mat_rev = np.zeros([2,3])
        div1 = mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0]
        mat_rev[0][0] = mat[1][1]/div1
        mat_rev[0][1] = -mat[0][1]/div1
        mat_rev[0][2] = -(mat[0][2]*mat[1][1]-mat[0][1]*mat[1][2])/div1
        div2 = mat[0][1]*mat[1][0]-mat[0][0]*mat[1][1]
        mat_rev[1][0] = mat[1][0]/div2
        mat_rev[1][1] = -mat[0][0]/div2
        mat_rev[1][2] = -(mat[0][2]*mat[1][0]-mat[0][0]*mat[1][2])/div2

        orisize = (oriimg.shape[1], oriimg.shape[0])
        target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)
        # source_image   = cv2.warpAffine(source_img, mat_rev, orisize)

        img_white = cv2.warpAffine(img_white, mat_rev, orisize)


        img_white[img_white>20] =255

        img_mask = img_white

        # if use_mask:
        #     kernel = np.ones((40,40),np.uint8)
        #     img_mask = cv2.erode(img_mask,kernel,iterations = 1)
        # else:
        kernel = np.ones((40,40),np.uint8)
        img_mask = cv2.erode(img_mask,kernel,iterations = 1)
        kernel_size = (20, 20)
        blur_size = tuple(2*i+1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)

        # kernel = np.ones((10,10),np.uint8)
        # img_mask = cv2.erode(img_mask,kernel,iterations = 1)



        img_mask /= 255

        img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])

        # pasing mask

        # target_image_parsing = postprocess(target_image, source_image, tgt_mask)

        target_image = np.array(target_image, dtype=np.float)[..., ::-1] * 255


        img_mask_list.append(img_mask)
        target_image_list.append(target_image)
        

        # target_image /= 255
        # target_image = 0
        img = np.array(oriimg, dtype=np.float)
        for img_mask, target_image in zip(img_mask_list, target_image_list):
            img = img_mask * target_image + (1-img_mask) * img
        
        final_img = img.astype(np.uint8)


        cv2.imwrite(opt.output_path + 'result.jpg',final_img)
        