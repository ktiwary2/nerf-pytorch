import torch
import efficient_shadow_mapping as eff_sm
from camera import Camera

######
# Sometimes loss goes to NaN :(
EPSILON = 1e-5
######



def efficient_sm_simple(cam_pixels, light_pixels, cam_depth, light_depth, 
                 single_ppc, light_ppc, image_shape, shadow_method):

    print(cam_pixels.shape, cam_depth.shape)
    mesh_range_cam = torch.cat([cam_pixels, cam_depth.view(-1,1)], dim=1)
    mesh_range_light = torch.cat([light_pixels, light_depth.view(-1,1)], dim=1)
    meshed_normed_light = eff_sm.get_normed_w(light_ppc, mesh_range_light, device=mesh_range_light.device)

    sm_fine = eff_sm.run_shadow_mapping(image_shape, single_ppc, light_ppc, mesh_range_cam, 
                                meshed_normed_light, mesh_range_cam.device, mode=shadow_method, \
                                delta=1e-2, epsilon=0.0, new_min=0.0, new_max=1.0, sigmoid=False, 
                                use_numpy_meshgrid=True)

    sm_fine = sm_fine.view(-1, 3)
    sm_fine = sm_fine + EPSILON * torch.ones_like(sm_fine)
    return sm_fine

def efficient_sm(cam_pixels, light_pixels, cam_results, light_results, 
                 ppc, light_ppc, image_shape, fine_sampling, Light_N_importance, shadow_method):
    """
    cam_pixels: [i,j,1]
    light_pixels: [i,j,1]
    cam_result: result dictionary with `depth_*`, `opacity_*`
    light_result: result dictionary with `depth_*`, `opacity_*`
    rays: generated rays 
    ppc: [Batch_size] Camera Poses: instance of the Camera() class
    light_ppc: [1] Pose of the Camera at Light position 
    batch_size: batch_size
    fine_sampling: set fine_sampling
    image_shape: IMAGE SHAPE OF THE CAMERA AT LIGHT POSITION 
    """

    def inference(ppc, light_camera, image_shape, batched_mesh_range_cam, meshed_normed_light_cam, shadow_method):
        """
        ppc: [Batch_size] Camera Poses: instance of the Camera() class
        light_camera: Instance of class Camera placed at the light position 
        batch_size: batch_size
        batched_mesh_range_cam: [num_rays, 4] (i, j, 1, depth] from camera viewpoints
        meshed_normed_light_cam: [H*W, 4] (i, j, 1, depth] from light viewpoint
        """
        shadow_maps = []
        curr_eye_pos = ppc['eye_pos'][0]
        prev_split_at = 0
        num_splits = 0 
        curr_ppc = Camera.from_camera_eyepos(eye_pos=ppc['eye_pos'][0].squeeze(0), camera=ppc['camera'][0].squeeze(0))
        for i in range(len(ppc['camera'])):
            # each pixel can have a different viewpoint within the batch, therefore we need to split them with the same (depth, pose) 
            # print("PPC: {} {}".format(curr_eye_pos, ppc['eye_pos'][i]))
            if not (torch.equal(curr_eye_pos, ppc['eye_pos'][i])): 
                # means a new ppc is encountered 
                # all pixels from prev_split to i have the same camera pose therefore we can estimate the sm 
                # for those.
                # print("------")
                # print("PPC: {} {}".format(curr_eye_pos, ppc['eye_pos'][i]))
                # print("Found different eye_pos, using shadow method {}, splitting at {}:{}".format(shadow_method, prev_split_at, i))
                sub_batch_mesh_range_cam = batched_mesh_range_cam[prev_split_at:i,:]
                sm = eff_sm.run_shadow_mapping(image_shape, curr_ppc, light_camera, sub_batch_mesh_range_cam, 
                                           meshed_normed_light_cam, sub_batch_mesh_range_cam.device, mode=shadow_method, \
                                           delta=1e-2, epsilon=0.0, new_min=0.0, new_max=1.0, sigmoid=False, 
                                           use_numpy_meshgrid=True)
                # print("sm, {}, {}".format(sm.shape, len(shadow_maps)))
                # print("------")
                shadow_maps += [sm]
                prev_split_at = i 
                curr_eye_pos = ppc['eye_pos'][i]
                curr_ppc = Camera.from_camera_eyepos(eye_pos=ppc['eye_pos'][i].squeeze(0), camera=ppc['camera'][i].squeeze(0))
                num_splits += 1

        if prev_split_at == 0: 
            # print("Found No Splits...")
            # means that all the pixels have the same viewpoint! we can batch them together
            shadow_maps = eff_sm.run_shadow_mapping(image_shape, curr_ppc, light_camera, batched_mesh_range_cam, 
                                        meshed_normed_light_cam, batched_mesh_range_cam.device, mode=shadow_method, \
                                        delta=1e-2, epsilon=0.0, new_min=0.0, new_max=1.0, sigmoid=False, 
                                        use_numpy_meshgrid=True) # (num_rays, 3)
        else: #prev_split_at < (len(ppc)-1): 
            # do inference on the remaining 
            # print("Doing inference on the last split from [{}:{}]".format(prev_split_at, len(ppc['camera'])))
            sub_batch_mesh_range_cam = batched_mesh_range_cam[prev_split_at:,:]
            sm = eff_sm.run_shadow_mapping(image_shape, curr_ppc, light_camera, sub_batch_mesh_range_cam, 
                                        meshed_normed_light_cam, sub_batch_mesh_range_cam.device, mode=shadow_method, \
                                        delta=1e-2, epsilon=0.0, new_min=0.0, new_max=1.0, sigmoid=False, 
                                        use_numpy_meshgrid=True) # (num_rays, 3)
            # print("sm, {}, {}".format(sm.shape, len(shadow_maps)))
            shadow_maps += [sm]
            shadow_maps = torch.cat(shadow_maps, 0) # (num_rays, 3)
            # print("shadow_maps.shape", shadow_maps.shape)

        if num_splits > 5:
            print("Split the batch of rays {} times. Not very efficient...".format(num_splits))
        return shadow_maps


    # should be the same for all the pixels! 
    # print(light_ppc)
    # print(type(light_ppc))
    # light_camera = Camera.from_camera_eyepos(eye_pos=light_ppc['eye_pos'].squeeze(0), camera=light_ppc['camera'].squeeze(0))
    light_camera = light_ppc
    if True: 
        # Do Shadow Mapping for Coarse Depth 
        cam_depths_coarse = cam_results['depth_coarse'] # (N_rays)
        cam_pixels = cam_pixels.to(cam_depths_coarse.device)
        batched_mesh_range_cam_coarse = torch.cat([cam_pixels, cam_depths_coarse.view(-1,1)], dim=1)
        # print(batched_mesh_range_cam_coarse)
        # assert N_rays == cam_depths_coarse.shape[0]
        # assert N_rays == light_depths_coarse.shape[0]

        light_depths_coarse = light_results['depth_coarse'] # (H*W)
        # This is not Batched, we do full inference on the light! 
        light_pixels = light_pixels.to(light_depths_coarse.device)
        mesh_range_light = torch.cat([light_pixels, light_depths_coarse.view(-1,1)], dim=1)
        # print(mesh_range_light.shape, mesh_range_light)
        meshed_normed_light_coarse = eff_sm.get_normed_w(light_camera, mesh_range_light, device=light_depths_coarse.device)
        # print("meshed_normed_light_coarse.shape", meshed_normed_light_coarse.shape)

        shadow_maps_coarse = inference(ppc, light_camera, image_shape, batched_mesh_range_cam_coarse, meshed_normed_light_coarse, shadow_method)
        # print("shadow_maps_coarse.shape", shadow_maps_coarse.shape)
        shadow_maps_coarse = shadow_maps_coarse.view(-1, 3)

        cam_results['sm_coarse'] = shadow_maps_coarse + EPSILON * torch.ones_like(shadow_maps_coarse) 

    # Do Shadow Mapping for Fine Depth Maps 
    FINE = True
    if fine_sampling and FINE: # sample points for fine model
        cam_depths_fine = cam_results['depth_fine'] # (N_rays)
        batched_mesh_range_cam_fine = torch.cat([cam_pixels, cam_depths_fine.view(-1,1)], dim=1)
        
        if Light_N_importance: 
            light_depths_fine = light_results['depth_fine'] # (N_rays)
            mesh_range_light = torch.cat([light_pixels, light_depths_fine.view(-1,1)], dim=1)
            meshed_normed_light_fine = eff_sm.get_normed_w(light_camera, mesh_range_light, device=light_depths_fine.device)

            shadow_maps_fine = inference(ppc, light_camera, image_shape, batched_mesh_range_cam_fine, meshed_normed_light_fine, shadow_method)
        else:
            shadow_maps_fine = inference(ppc, light_camera, image_shape, batched_mesh_range_cam_fine, meshed_normed_light_coarse, shadow_method)

        # print("shadow_maps_fine.shape", shadow_maps_fine.shape)
        shadow_maps_fine = shadow_maps_fine.view(-1, 3)
        cam_results['sm_fine'] = shadow_maps_fine + EPSILON * torch.ones_like(shadow_maps_coarse)


    return cam_results
