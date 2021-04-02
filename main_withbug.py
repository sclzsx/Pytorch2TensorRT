import os
from tqdm import tqdm
from pathlib import Path
import onnxruntime
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 非常重要，不可删除
import time
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
from collections import OrderedDict

from SXNet import SXNet
from utils import *
from nets.SSD_VGG_Optim_FPN_RFB import build_net

cfg = VOC_300
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()


def pre_process(img, gpu=True):
    transform = BaseTransform(input_size, rgb_means, (2, 0, 1))
    with torch.no_grad():
        x = transform(img).unsqueeze(0)
        if gpu:
            x = x.cuda()
    return x


def post_process(img, y, conf_thresh=0.5, nms_iou_th=0.45, dst_size=(640, 480)):
    if img.shape[:2] != [dst_size[1], dst_size[0]]:
        img = cv2.resize(img, dst_size)
    out = (y[:, :, :4], y[:, :, 4:].squeeze(0))
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).cuda()
    detector = Detect(num_classes, 0, cfg)
    boxes, scores = detector.forward(out, priors)
    boxes = boxes[0]
    scores = scores[0]
    boxes *= scale
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    all_boxes = [[] for _ in range(num_classes)]
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > conf_thresh)[0]
        if len(inds) == 0:
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(c_dets, nms_iou_th)
        c_dets = c_dets[keep, :]
        all_boxes[j] = c_dets
    for k in range(1, num_classes):
        if len(all_boxes[k]) > 0:
            box_scores = all_boxes[k][:, 4]
            box_locations = all_boxes[k][:, 0:4]
            for i, box_location in enumerate(box_locations):
                p1 = (int(box_location[0]), int(box_location[1]))
                p2 = (int(box_location[2]), int(box_location[3]))
                cv2.rectangle(img, p1, p2, COLORS_BGR[k], 2)
                title = "%s:%.2f" % (VOC_CLASSES[k], box_scores[i])
                cv2.rectangle(img, (p1[0] - 1, p1[1] - 1), (p2[0] + 1, p1[1] + 20), COLORS_BGR[k], -1)
                p3 = (p1[0] + 2, p1[1] + 15)
                cv2.putText(img, title, p3, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    return img


def onnx_to_tensorrt(onnx_path, mode, calib=None):
    save_trt_path = onnx_path.replace('.onnx', '')
    save_trt_path = save_trt_path + '-' + mode + '.trt'
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(network_flags) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        if mode == 'fp16':
            builder.fp16_mode = True
        elif mode == 'fp32':
            builder.fp16_mode = False
        elif mode == 'int8':
            builder.int8_mode = True
            builder.int8_calibrator = calib
        else:
            print('Unsupported mode!')
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            parser.parse(model.read())
        engine = builder.build_cuda_engine(network)
        with open(save_trt_path, 'wb') as f:
            f.write(engine.serialize())


def load_net_dn(dn_pt_path):
    if 'SXNet' in dn_pt_path:
        net_name = dn_pt_path.split('-')[0]
        info = net_name.split('_')
        basic_ch, rfb_ch, asy, tlu, bias = int(info[1]), int(info[2]), False, False, False
        if '_a' in net_name:
            asy = True
        if '_t' in net_name:
            tlu = True
        if '_b' in net_name:
            bias = True
        net = SXNet(basic_ch, rfb_ch, asy=asy, tlu=tlu, bias=bias, phase='test')
    state_dict = torch.load(dn_pt_path)
    net.load_state_dict(state_dict)
    net.eval()
    net.cuda()
    return net


def load_net_dt(img_dim, trained_model):
    net = build_net('test', img_dim, num_classes)
    state_dict = torch.load(trained_model)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    net = net.cuda()
    cudnn.benchmark = True
    return net


def torch_to_trt(full_pt_net, exp_tensor, onnx_path, trt_mode):
    _ = torch.onnx.export(full_pt_net, exp_tensor, onnx_path, verbose=False, training=False,
                          do_constant_folding=True, input_names=['input'], output_names=['output'])
    onnx_to_tensorrt(onnx_path, trt_mode)


# def predict_video_pt_trt(vid_path, model_path, denoise_model_path, dst_path):
#     denoise_model = load_net_dn(denoise_model_path)
#     G_LOGGER = trt.Logger(trt.Logger.ERROR)  # LOG提示的级别是出错才提示
#     with trt.Runtime(G_LOGGER) as runtime:
#         with open(model_path, "rb") as f_dt:
#             eng_dt = runtime.deserialize_cuda_engine(f_dt.read())
#             for binding in eng_dt:
#                 if not eng_dt.binding_is_input(binding):
#                     out_shape_dt = eng_dt.get_binding_shape(binding)
#             with eng_dt.create_execution_context() as cont_dt:
#                 if os.path.exists(vid_path):
#                     dst_path = dst_path.replace('.flv', '_pt_trt.flv')
#                     print('save_vid_path:', dst_path)
#                     vw = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*'FLV1'), 12, (640 * 2, 480))
#                     cap = cv2.VideoCapture(vid_path)
#                     fps, psnr = [], []
#                 else:
#                     print('processing camera!')
#                     cap = cv2.VideoCapture(0)
#                 while cap.isOpened():
#                     ret, clean = cap.read()
#                     if not ret:
#                         break
#                     noised = add_multi_noise(clean, 10)
#                     s = time.clock()
#                     if denoise_model_path is not None:
#                         x = torch.from_numpy(np.float32(noised / 255)).permute(2, 0, 1).unsqueeze(0).cuda()
#                         with torch.no_grad():
#                             y = denoise_model(x)
#                         xx = torch.from_numpy(np.float32(clean / 255)).permute(2, 0, 1).unsqueeze(0).cuda()
#                         psnr_ = batch_PSNR(y, xx, data_range=1.0)
#                         psnr.append(psnr_)
#                         img = np.array(y.cpu().squeeze(0).permute(1, 2, 0) * 255).astype('uint8')
#                     else:
#                         img = noised
#                         psnr_ = 0
#                     psnr.append(psnr_)
#                     x2 = pre_process(img, gpu=False)
#                     dn_fps = np.round(1 / (time.clock() - s), 3)
#                     ss = time.clock()
#                     image = np.ascontiguousarray(x2)
#                     d_input = cuda.mem_alloc(1 * image.size * image.dtype.itemsize)
#                     y = np.empty(out_shape_dt, dtype=np.float32)
#                     d_output = cuda.mem_alloc(1 * y.size * y.dtype.itemsize)
#                     bindings = [int(d_input), int(d_output)]
#                     stream = cuda.Stream()
#                     cuda.memcpy_htod_async(d_input, image, stream)
#                     cont_dt.execute_async(int(1), bindings, stream.handle, None)
#                     cuda.memcpy_dtoh_async(y, d_output, stream)
#                     stream.synchronize()
#                     y = torch.from_numpy(y).cuda()
#                     img = post_process(img, y)
#                     dt_fps = np.round(1 / (time.clock() - ss), 3)
#                     fps_ = np.round(1 / (time.clock() - s), 3)
#                     fps.append(fps_)
#                     print('FPS:{}, PSNR:{}, Dn_fps:{}, Det_fps:{}'.format(fps_, psnr_, dn_fps, dt_fps))
#                     cv2.putText(img, 'FPS:   ' + str(fps_), (10, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
#                     cv2.putText(img, 'PSNR:  ' + str(psnr_), (10, 50), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
#                     cv2.putText(img, 'DnFPS: ' + str(dn_fps), (10, 80), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
#                     cv2.putText(img, 'DtFPS: ' + str(dt_fps), (10, 110), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
#                     img = cv2.hconcat([noised, img])
#                     if os.path.exists(vid_path):
#                         vw.write(img)
#                     else:
#                         cv2.imshow('dn and dt', img)
#                         cv2.waitKey(1)
#                 print('m_fps:{}, m_psnr:{}'.format(np.mean(fps), np.mean(psnr)))


# def predict_video_pt_pt(vid_path, model_path, denoise_model_path, dst_path):
#     denoise_model = load_net_dn(denoise_model_path)
#     G_LOGGER = trt.Logger(trt.Logger.ERROR)  # LOG提示的级别是出错才提示
#     with trt.Runtime(G_LOGGER) as runtime:
#         with open(model_path, "rb") as f_dt:
#             eng_dt = runtime.deserialize_cuda_engine(f_dt.read())
#             for binding in eng_dt:
#                 if not eng_dt.binding_is_input(binding):
#                     out_shape_dt = eng_dt.get_binding_shape(binding)
#             with eng_dt.create_execution_context() as cont_dt:
#                 if os.path.exists(vid_path):
#                     dst_path = dst_path.replace('.flv', '_pt_trt.flv')
#                     print('save_vid_path:', dst_path)
#                     vw = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*'FLV1'), 12, (640 * 2, 480))
#                     cap = cv2.VideoCapture(vid_path)
#                     fps, psnr = [], []
#                 else:
#                     print('processing camera!')
#                     cap = cv2.VideoCapture(0)
#                 while cap.isOpened():
#                     ret, clean = cap.read()
#                     if not ret:
#                         break
#                     noised = add_multi_noise(clean, 10)
#                     s = time.clock()
#                     if denoise_model_path is not None:
#                         x = torch.from_numpy(np.float32(noised / 255)).permute(2, 0, 1).unsqueeze(0).cuda()
#                         with torch.no_grad():
#                             y = denoise_model(x)
#                         xx = torch.from_numpy(np.float32(clean / 255)).permute(2, 0, 1).unsqueeze(0).cuda()
#                         psnr_ = batch_PSNR(y, xx, data_range=1.0)
#                         psnr.append(psnr_)
#                         img = np.array(y.cpu().squeeze(0).permute(1, 2, 0) * 255).astype('uint8')
#                     else:
#                         img = noised
#                         psnr_ = 0
#                     psnr.append(psnr_)
#                     x2 = pre_process(img, gpu=False)
#                     dn_fps = np.round(1 / (time.clock() - s), 3)
#                     ss = time.clock()
#                     image = np.ascontiguousarray(x2)
#                     d_input = cuda.mem_alloc(1 * image.size * image.dtype.itemsize)
#                     y = np.empty(out_shape_dt, dtype=np.float32)
#                     d_output = cuda.mem_alloc(1 * y.size * y.dtype.itemsize)
#                     bindings = [int(d_input), int(d_output)]
#                     stream = cuda.Stream()
#                     cuda.memcpy_htod_async(d_input, image, stream)
#                     cont_dt.execute_async(int(1), bindings, stream.handle, None)
#                     cuda.memcpy_dtoh_async(y, d_output, stream)
#                     stream.synchronize()
#                     y = torch.from_numpy(y).cuda()
#                     img = post_process(img, y)
#                     dt_fps = np.round(1 / (time.clock() - ss), 3)
#                     fps_ = np.round(1 / (time.clock() - s), 3)
#                     fps.append(fps_)
#                     print('FPS:{}, PSNR:{}, Dn_fps:{}, Det_fps:{}'.format(fps_, psnr_, dn_fps, dt_fps))
#                     cv2.putText(img, 'FPS:   ' + str(fps_), (10, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
#                     cv2.putText(img, 'PSNR:  ' + str(psnr_), (10, 50), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
#                     cv2.putText(img, 'DnFPS: ' + str(dn_fps), (10, 80), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
#                     cv2.putText(img, 'DtFPS: ' + str(dt_fps), (10, 110), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
#                     img = cv2.hconcat([noised, img])
#                     if os.path.exists(vid_path):
#                         vw.write(img)
#                     else:
#                         cv2.imshow('dn and dt', img)
#                         cv2.waitKey(1)
#                 print('m_fps:{}, m_psnr:{}'.format(np.mean(fps), np.mean(psnr)))


# def predict_video_trt_trt(vid_path, dn_pt_path, dt_pt_path, save_dir, convert_flag):
#     dn_trt_path = dn_pt_path.replace('.pth', '-fp16.trt')
#     if not os.path.exists(dn_trt_path) or convert_flag:
#         print('convert to', dn_trt_path)
#         onnx_path = dn_pt_path.replace('.pth', '.onnx')
#         full_pt_net = load_net_dn(dn_pt_path)
#         with torch.no_grad():
#             exp_tensor = torch.randn(1, 3, 480, 640).cuda()
#             trt_mode = dn_trt_path.split('-')[-1].split('.')[0]
#             torch_to_trt(full_pt_net, exp_tensor, onnx_path, trt_mode)
#
#     dt_trt_path = dt_pt_path.replace('.pth', '-fp16.trt')
#     if not os.path.exists(dt_trt_path) or convert_flag:
#         print('convert to', dt_trt_path)
#         onnx_path = dt_pt_path.replace('.pth', '.onnx')
#         full_pt_net = load_net_dt(300, dt_pt_path)
#         with torch.no_grad():
#             exp_tensor = torch.randn(1, 3, 300, 300).cuda()
#             trt_mode = dt_trt_path.split('-')[-1].split('.')[0]
#             torch_to_trt(full_pt_net, exp_tensor, onnx_path, trt_mode)
#
#     print('loading:', dn_trt_path, dt_trt_path)
#     G_LOGGER = trt.Logger(trt.Logger.ERROR)  # LOG提示的级别是出错才提示
#     with trt.Runtime(G_LOGGER) as runtime:
#         with open(dn_trt_path, "rb") as f_dn:
#             eng_dn = runtime.deserialize_cuda_engine(f_dn.read())
#             for binding in eng_dn:
#                 if not eng_dn.binding_is_input(binding):
#                     out_shape_dn = eng_dn.get_binding_shape(binding)
#             with eng_dn.create_execution_context() as cont_dn:
#                 with open(dt_trt_path, "rb") as f_dt:
#                     eng_dt = runtime.deserialize_cuda_engine(f_dt.read())
#                     for binding in eng_dt:
#                         if not eng_dt.binding_is_input(binding):
#                             out_shape_dt = eng_dt.get_binding_shape(binding)
#                     with eng_dt.create_execution_context() as cont_dt:
#                         if os.path.exists(vid_path):
#                             print('predicting_video:', vid_path)
#                             save_vid_path = save_dir + Path(dn_trt_path).name + Path(dt_trt_path).name + '.flv'
#                             print('save_vid_path:', save_vid_path)
#                             vw = cv2.VideoWriter(save_vid_path, cv2.VideoWriter_fourcc(*'FLV1'), 18, (640 * 2, 480))
#                             cap = cv2.VideoCapture(vid_path)
#                             fps, psnr = [], []
#                         else:
#                             print('processing camera!')
#                             cap = cv2.VideoCapture(0)
#                         while cap.isOpened():
#                             ret, clean = cap.read()
#                             if not ret:
#                                 break
#                             noised = add_multi_noise(clean, 10)
#                             s = time.clock()
#                             if dn_trt_path is not None:
#                                 x = torch.from_numpy(np.float32(noised / 255)).permute(2, 0, 1).unsqueeze(0)
#                                 xx = torch.from_numpy(np.float32(clean / 255)).permute(2, 0, 1).unsqueeze(0)
#                                 image = np.ascontiguousarray(x)
#                                 d_input = cuda.mem_alloc(1 * image.size * image.dtype.itemsize)
#                                 y = np.empty(out_shape_dn, dtype=np.float32)
#                                 d_output = cuda.mem_alloc(1 * y.size * y.dtype.itemsize)
#                                 bindings = [int(d_input), int(d_output)]
#                                 stream = cuda.Stream()
#                                 cuda.memcpy_htod_async(d_input, image, stream)
#                                 cont_dn.execute_async(int(1), bindings, stream.handle, None)
#                                 cuda.memcpy_dtoh_async(y, d_output, stream)
#                                 stream.synchronize()
#                                 y = torch.from_numpy(y)
#                                 psnr_ = np.round(batch_PSNR(y, xx, data_range=1.0), 3)
#                                 img = np.array(y.cpu().squeeze(0).permute(1, 2, 0) * 255).astype('uint8')
#                             else:
#                                 img = noised
#                                 psnr_ = 0
#                             psnr.append(psnr_)
#                             x = pre_process(img, gpu=False)
#                             dn_fps = np.round(1 / (time.clock() - s), 3)
#                             ss = time.clock()
#                             image = np.ascontiguousarray(x)
#                             d_input = cuda.mem_alloc(1 * image.size * image.dtype.itemsize)
#                             y = np.empty(out_shape_dt, dtype=np.float32)
#                             d_output = cuda.mem_alloc(1 * y.size * y.dtype.itemsize)
#                             bindings = [int(d_input), int(d_output)]
#                             stream = cuda.Stream()
#                             cuda.memcpy_htod_async(d_input, image, stream)
#                             cont_dt.execute_async(int(1), bindings, stream.handle, None)
#                             cuda.memcpy_dtoh_async(y, d_output, stream)
#                             stream.synchronize()
#                             y = torch.from_numpy(y).cuda()
#                             img = post_process(img, y)
#                             dt_fps = np.round(1 / (time.clock() - ss), 3)
#                             fps_ = np.round(1 / (time.clock() - s), 3)
#                             fps.append(fps_)
#                             print('FPS:{}, PSNR:{}, Dn_fps:{}, Det_fps:{}'.format(fps_, psnr_, dn_fps, dt_fps))
#                             cv2.putText(img, 'FPS:   ' + str(fps_), (10, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
#                             cv2.putText(img, 'PSNR:  ' + str(psnr_), (10, 50), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
#                             cv2.putText(img, 'DnFPS: ' + str(dn_fps), (10, 80), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
#                             cv2.putText(img, 'DtFPS: ' + str(dt_fps), (10, 110), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
#                             img = cv2.hconcat([noised, img])
#                             if os.path.exists(vid_path):
#                                 vw.write(img)
#                             else:
#                                 cv2.imshow('dn and dt', img)
#                                 cv2.waitKey(1)
#                         print('m_fps:{}, m_psnr:{}'.format(np.mean(fps), np.mean(psnr)))
def predict_image_dn_trt(noised, clean, out_shape, context):
    x = torch.from_numpy(np.float32(noised / 255)).permute(2, 0, 1).unsqueeze(0)
    image = np.ascontiguousarray(x)
    d_input = cuda.mem_alloc(1 * image.size * image.dtype.itemsize)
    y = np.empty(out_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(1 * y.size * y.dtype.itemsize)
    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_input, image, stream)
    context.execute_async(int(1), bindings, stream.handle, None)
    cuda.memcpy_dtoh_async(y, d_output, stream)
    stream.synchronize()
    y = torch.from_numpy(y)
    if clean is not None:
        xx = torch.from_numpy(np.float32(clean / 255)).permute(2, 0, 1).unsqueeze(0)
        psnr_ = np.round(batch_PSNR(y, xx, data_range=1.0), 3)
    else:
        psnr_ = 0
    img = np.array(y.cpu().squeeze(0).permute(1, 2, 0) * 255).astype('uint8')
    return psnr_, img


def predict_image_dt_trt(in_img, out_shape, context):
    x = pre_process(in_img, gpu=False)
    image = np.ascontiguousarray(x)
    d_input = cuda.mem_alloc(1 * image.size * image.dtype.itemsize)
    y = np.empty(out_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(1 * y.size * y.dtype.itemsize)
    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_input, image, stream)
    context.execute_async(int(1), bindings, stream.handle, None)
    cuda.memcpy_dtoh_async(y, d_output, stream)
    stream.synchronize()
    y = torch.from_numpy(y).cuda()
    out_img = post_process(in_img, y)
    return out_img


def predict_image_dt_pt(in_img, net):
    x = pre_process(in_img, gpu=True)
    with torch.no_grad():
        y = net(x)
    out_img = post_process(in_img, y)
    return out_img


def predict_image_dn_pt(source, GT, net):
    x = torch.from_numpy(np.float32(source / 255)).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        y = net(x)
        if GT is not None:
            xx = torch.from_numpy(np.float32(GT / 255)).permute(2, 0, 1).unsqueeze(0).cuda()
            psnr_ = np.round(batch_PSNR(y, xx, data_range=1.0), 3)
        else:
            psnr_ = 0
    img = np.array(y.cpu().squeeze(0).permute(1, 2, 0) * 255).astype('uint8')
    return psnr_, img


def prepare_dn_trt(dn_pt_path, convert_flag):
    dn_trt_path = dn_pt_path.replace('.pth', '-fp16.trt')
    if not os.path.exists(dn_trt_path) or convert_flag:
        print('convert to', dn_trt_path)
        onnx_path = dn_pt_path.replace('.pth', '.onnx')
        full_pt_net = load_net_dn(dn_pt_path)
        with torch.no_grad():
            exp_tensor = torch.randn(1, 3, 480, 640).cuda()
            trt_mode = dn_trt_path.split('-')[-1].split('.')[0]
            torch_to_trt(full_pt_net, exp_tensor, onnx_path, trt_mode)
    print(dn_trt_path, 'Ready!')
    return dn_trt_path


def prepare_dt_trt(dt_pt_path, convert_flag):
    dt_trt_path = dt_pt_path.replace('.pth', '-fp16.trt')
    if not os.path.exists(dt_trt_path) or convert_flag:
        print('convert to', dt_trt_path)
        onnx_path = dt_pt_path.replace('.pth', '.onnx')
        full_pt_net = load_net_dt(300, dt_pt_path)
        with torch.no_grad():
            exp_tensor = torch.randn(1, 3, 300, 300).cuda()
            trt_mode = dt_trt_path.split('-')[-1].split('.')[0]
            torch_to_trt(full_pt_net, exp_tensor, onnx_path, trt_mode)
    print(dt_trt_path, 'Ready!')
    return dt_trt_path


def predict_video_trt(vid_path, pt_path, mode, save_dir, convert_flag):
    if 'dn' in mode:
        trt_path = prepare_dn_trt(pt_path, convert_flag)
    elif mode == 'dt':
        trt_path = prepare_dt_trt(pt_path, convert_flag)
    else:
        print('Unsupported mode')

    G_LOGGER = trt.Logger(trt.Logger.ERROR)  # LOG提示的级别是出错才提示
    with trt.Runtime(G_LOGGER) as runtime:
        with open(trt_path, "rb") as f:
            eng = runtime.deserialize_cuda_engine(f.read())
            for binding in eng:
                if not eng.binding_is_input(binding):
                    out_shape = eng.get_binding_shape(binding)
            with eng.create_execution_context() as cont:

                if os.path.exists(vid_path):
                    print('predicting_video:', vid_path)
                    save_vid_path = save_dir + Path(trt_path).name + '.flv'
                    print('save_vid_path:', save_vid_path)
                    vw = cv2.VideoWriter(save_vid_path, cv2.VideoWriter_fourcc(*'FLV1'), 18, (640 * 2, 480))
                    cap = cv2.VideoCapture(vid_path)
                    fps, psnr = [], []
                else:
                    print('processing camera!')
                    cap = cv2.VideoCapture(0)
                while cap.isOpened():
                    if 'real' in mode:
                        ret, source = cap.read()
                        if not ret:
                            break
                        GT = None
                    else:
                        ret, GT = cap.read()
                        if not ret:
                            break
                        if 'dn' in mode:
                            source = add_multi_noise(GT, 10)

                    s1 = time.clock()

                    if mode == 'dn':
                        psnr_, out_img = predict_image_dn_trt(source, GT, out_shape, cont)
                        fps_ = np.round(1 / (time.clock() - s1), 3)
                        psnr.append(psnr_)
                        fps.append(fps_)
                        print('FPS:{}, PSNR:{}'.format(fps_, psnr_))
                        img = out_img.copy()
                        cv2.putText(img, 'FPS:   ' + str(fps_), (10, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
                        cv2.putText(img, 'PSNR:  ' + str(psnr_), (10, 50), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
                        img = cv2.hconcat([source, img])

                    elif mode == 'dt':
                        out_img = predict_image_dt_trt(source, out_shape, cont)
                        fps_ = np.round(1 / (time.clock() - s1), 3)
                        psnr_ = 0
                        psnr.append(psnr_)
                        fps.append(fps_)
                        print('FPS:{}, PSNR:{}'.format(fps_, psnr_))
                        img = out_img.copy()
                        cv2.putText(img, 'FPS:   ' + str(fps_), (10, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
                        cv2.putText(img, 'PSNR:  ' + str(psnr_), (10, 50), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)

                    if os.path.exists(vid_path):
                        plt.imshow(img)
                        plt.pause(0.5)
                        vw.write(img)
                    else:
                        cv2.imshow('dn and dt', img)
                        cv2.waitKey(1)
                print('m_fps:{}, m_psnr:{}'.format(np.mean(fps), np.mean(psnr)))


def predict_video_pt(vid_path, pt_path, mode, save_dir):
    if 'dn' in mode:
        net = load_net_dn(pt_path)
    elif mode == 'dt':
        net = load_net_dt(300, dt_pt_path)
    else:
        print('Unsupported mode')

    if os.path.exists(vid_path):
        print('predicting_video:', vid_path)
        save_vid_path = save_dir + Path(pt_path).name + '.flv'
        print('save_vid_path:', save_vid_path)
        vw = cv2.VideoWriter(save_vid_path, cv2.VideoWriter_fourcc(*'FLV1'), 18, (640 * 2, 480))
        cap = cv2.VideoCapture(vid_path)
        fps, psnr = [], []
    else:
        print('processing camera!')
        cap = cv2.VideoCapture(0)
    while cap.isOpened():
        if 'real' in mode:
            ret, source = cap.read()
            if not ret:
                break
            GT = None
        else:
            ret, GT = cap.read()
            if not ret:
                break
            if 'dn' in mode:
                source = add_multi_noise(GT, 10)

        s1 = time.clock()

        if mode == 'dn':
            psnr_, out_img = predict_image_dn_pt(source, GT, net)
            fps_ = np.round(1 / (time.clock() - s1), 3)
            psnr.append(psnr_)
            fps.append(fps_)
            print('FPS:{}, PSNR:{}'.format(fps_, psnr_))
            img = out_img.copy()
            cv2.putText(img, 'FPS:   ' + str(fps_), (10, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
            cv2.putText(img, 'PSNR:  ' + str(psnr_), (10, 50), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
            img = cv2.hconcat([source, img])

        elif mode == 'dt':
            out_img = predict_image_dt_pt(GT, net)
            fps_ = np.round(1 / (time.clock() - s1), 3)
            psnr_ = 0
            psnr.append(psnr_)
            fps.append(fps_)
            print('FPS:{}, PSNR:{}'.format(fps_, psnr_))
            img = out_img.copy()
            cv2.putText(img, 'FPS:   ' + str(fps_), (10, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
            cv2.putText(img, 'PSNR:  ' + str(psnr_), (10, 50), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)

        if os.path.exists(vid_path):
            vw.write(img)
        else:
            cv2.imshow('dn and dt', img)
            cv2.waitKey(1)
    print('m_fps:{}, m_psnr:{}'.format(np.mean(fps), np.mean(psnr)))


def predict_video_trt_trt(vid_path, mode, dn_pt_path, dt_pt_path, save_dir, convert_flag):
    dn_trt_path = prepare_dn_trt(dn_pt_path, convert_flag)
    dt_trt_path = prepare_dt_trt(dt_pt_path, convert_flag)

    G_LOGGER = trt.Logger(trt.Logger.ERROR)  # LOG提示的级别是出错才提示
    with trt.Runtime(G_LOGGER) as runtime:
        with open(dn_trt_path, "rb") as f_dn:
            eng_dn = runtime.deserialize_cuda_engine(f_dn.read())
            for binding in eng_dn:
                if not eng_dn.binding_is_input(binding):
                    out_shape_dn = eng_dn.get_binding_shape(binding)
            with eng_dn.create_execution_context() as cont_dn:
                with open(dt_trt_path, "rb") as f_dt:
                    eng_dt = runtime.deserialize_cuda_engine(f_dt.read())
                    for binding in eng_dt:
                        if not eng_dt.binding_is_input(binding):
                            out_shape_dt = eng_dt.get_binding_shape(binding)
                    with eng_dt.create_execution_context() as cont_dt:

                        if os.path.exists(vid_path):
                            print('predicting_video:', vid_path)
                            save_vid_path = save_dir + Path(dn_trt_path).name + Path(dt_trt_path).name + '.flv'
                            print('save_vid_path:', save_vid_path)
                            vw = cv2.VideoWriter(save_vid_path, cv2.VideoWriter_fourcc(*'FLV1'), 18, (640 * 2, 480))
                            cap = cv2.VideoCapture(vid_path)
                            fps, psnr = [], []
                        else:
                            print('processing camera!')
                            cap = cv2.VideoCapture(0)
                        while cap.isOpened():
                            if 'real' in mode:
                                ret, source = cap.read()
                                if not ret:
                                    break
                                h, w = source.shape[:2]
                                source = source[h // 2 - 240:h // 2 + 240, w // 2 - 320:w // 2 + 320]
                                GT = None
                            else:
                                ret, GT = cap.read()
                                if not ret:
                                    break
                                h, w = GT.shape[:2]
                                GT = GT[h // 2 - 240:h // 2 + 240, w // 2 - 320:w // 2 + 320]
                                if 'dn' in mode:
                                    source = add_multi_noise(GT, 10)
                            s1 = time.clock()

                            psnr_, img = predict_image_dn_trt(source, GT, out_shape_dn, cont_dn)
                            dn_fps = np.round(1 / (time.clock() - s1), 3)
                            psnr.append(psnr_)

                            s2 = time.clock()
                            img = predict_image_dt_trt(img, out_shape_dt, cont_dt)
                            dt_fps = np.round(1 / (time.clock() - s2), 3)
                            fps_ = np.round(1 / (time.clock() - s1), 3)
                            fps.append(fps_)

                            print('FPS:{}, PSNR:{}, Dn_fps:{}, Det_fps:{}'.format(fps_, psnr_, dn_fps, dt_fps))
                            cv2.putText(img, 'FPS:   ' + str(fps_), (10, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
                            cv2.putText(img, 'PSNR:  ' + str(psnr_), (10, 50), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
                            cv2.putText(img, 'DnFPS: ' + str(dn_fps), (10, 80), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
                            cv2.putText(img, 'DtFPS: ' + str(dt_fps), (10, 110), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
                            img = cv2.hconcat([source, img])

                            if os.path.exists(vid_path):
                                plt.imshow(img)
                                plt.pause(0.5)
                                vw.write(img)
                            else:
                                cv2.imshow('dn and dt', img)
                                cv2.waitKey(1)
                        print('m_fps:{}, m_psnr:{}'.format(np.mean(fps), np.mean(psnr)))


def predict_video_pt_pt(vid_path, mode, dn_pt_path, dt_pt_path, save_dir):
    net_dn = load_net_dn(dn_pt_path)
    net_dt = load_net_dt(300, dt_pt_path)

    if os.path.exists(vid_path):
        print('predicting_video:', vid_path)
        save_vid_path = save_dir + Path(dn_pt_path).name + Path(dt_pt_path).name + '.flv'
        print('save_vid_path:', save_vid_path)
        vw = cv2.VideoWriter(save_vid_path, cv2.VideoWriter_fourcc(*'FLV1'), 18, (640 * 2, 480))
        cap = cv2.VideoCapture(vid_path)
        fps, psnr = [], []
    else:
        print('processing camera!')
        cap = cv2.VideoCapture(0)
    while cap.isOpened():
        if 'real' in mode:
            ret, source = cap.read()
            if not ret:
                break
            h, w = source.shape[:2]
            source = source[h // 2 - 240:h // 2 + 240, w // 2 - 320:w // 2 + 320]
            GT = None
        else:
            ret, GT = cap.read()
            if not ret:
                break
            h, w = GT.shape[:2]
            GT = GT[h // 2 - 240:h // 2 + 240, w // 2 - 320:w // 2 + 320]
            if 'dn' in mode:
                source = add_multi_noise(GT, 10)

        s1 = time.clock()

        psnr_, img = predict_image_dn_pt(source, GT, net_dn)

        dn_fps = np.round(1 / (time.clock() - s1), 3)
        psnr.append(psnr_)

        s2 = time.clock()
        img = predict_image_dt_pt(img, net_dt)
        dt_fps = np.round(1 / (time.clock() - s2), 3)
        fps_ = np.round(1 / (time.clock() - s1), 3)
        fps.append(fps_)

        print('FPS:{}, PSNR:{}, Dn_fps:{}, Det_fps:{}'.format(fps_, psnr_, dn_fps, dt_fps))
        cv2.putText(img, 'FPS:   ' + str(fps_), (10, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
        cv2.putText(img, 'PSNR:  ' + str(psnr_), (10, 50), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
        cv2.putText(img, 'DnFPS: ' + str(dn_fps), (10, 80), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
        cv2.putText(img, 'DtFPS: ' + str(dt_fps), (10, 110), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
        img = cv2.hconcat([source, img])

        if os.path.exists(vid_path):
            # plt.imshow(img)
            # plt.pause(0.5)
            vw.write(img)
        else:
            cv2.imshow('dn and dt', img)
            cv2.waitKey(1)
    print('m_fps:{}, m_psnr:{}'.format(np.mean(fps), np.mean(psnr)))


if __name__ == '__main__':
    save_dir = 'results/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    modes = {0: 'dn_dt_real_pt', 1: 'dn_dt_real_trt',
             2: 'dn_dt_stim_trt', 3: 'dn_dt_stim_pt',
             4: 'dn_real_trt', 5: 'dt_real_trt',
             6: 'dn_stim_pt', 7: 'dt_stim_pt'}

    dt_pt_path = 'files/300.pth'
    convert_flag = False

    vid_path = 'files/nuclear_real_cropped.avi'
    dn_pt_path = 'files/SXNet_8_16_a-nuclear_real.pth'
    mode_id = 1

    if mode_id == 0:
        predict_video_pt_pt(vid_path, modes[mode_id], dn_pt_path, dt_pt_path, save_dir)
    elif mode_id == 1:
        predict_video_trt_trt(vid_path, modes[mode_id], dn_pt_path, dt_pt_path, save_dir, convert_flag)
    elif mode_id == 2:
        predict_video_trt(vid_path, modes[mode_id], dn_pt_path, save_dir, convert_flag)
    elif mode_id == 3:
        predict_video_trt(vid_path, modes[mode_id], dt_pt_path, save_dir, convert_flag)
    elif mode_id == 4:
        predict_video_pt(vid_path, modes[mode_id], dn_pt_path, save_dir)
    elif mode_id == 5:
        predict_video_pt(vid_path, modes[mode_id], dt_pt_path, save_dir)
    elif mode_id == 6:
        predict_video_pt(vid_path, modes[mode_id], dn_pt_path, save_dir)
    elif mode_id == 7:
        predict_video_pt(vid_path, modes[mode_id], dt_pt_path, save_dir)
