def _get_smpl_data_huge100k(self, smpl_params, camera):

    betas=smpl_params['betas'].to(self.device)
    global_orient=smpl_params['global_orient'].to(self.device)
    body_pose=smpl_params['body_pose'].to(self.device)
    left_hand_pose=smpl_params['left_hand_pose'].to(self.device)
    right_hand_pose=smpl_params['right_hand_pose'].to(self.device)
    jaw_pose=smpl_params['jaw_pose'].to(self.device)
    leye_pose=smpl_params['leye_pose'].to(self.device)
    reye_pose=smpl_params['reye_pose'].to(self.device)
    expression=smpl_params['expression'].to(self.device)

    transl = smpl_params['transl'].to(self.device)
    scale = smpl_params['scale'].to(self.device)


    smpl_verts = ((smpl_out.vertices[0] * param['scale'])).detach()

    body = self.body_model(global_orient=global_orient, body_pose=body_pose, 
                            betas=betas, transl=transl,
                            left_hand_pose=left_hand_pose,
                            right_hand_pose=right_hand_pose, jaw_pose=jaw_pose, 
                            leye_pose=leye_pose, reye_pose=reye_pose,
                            expression=expression)
    
    transforms_mat = body.transforms_mat[0]

    # get relative rotation matrix
    poses_all = body.full_pose
    poses_all = Rotation.from_rotvec(poses_all.reshape(-1, 3).cpu().numpy()).as_matrix() # 55 x 3 x 3
    poses_all = torch.from_numpy(poses_all.reshape([-1, 9])).to(torch.float32)  # 55 x 9, including root rotation
    poses_all = poses_all.to(self.device)


    # get transformation matrix from rest pose to vitruvian pose
    body = self.body_model(betas=betas.to(self.device), expression=expression.to(self.device))
    vertices_rest = body.vertices.detach()[0]
    self.get_cano_smpl_verts(vertices_rest, camera)

    J_regressor = self.body_model.J_regressor 
    Jtr = torch.matmul(J_regressor, vertices_rest)
    transforms_mat_02v = get_02v_bone_transforms(Jtr)

    # final bone transforms that transforms the canonical Vitruvian-pose mesh to the posed mesh
    transforms_mat_02v_inv = torch.linalg.inv(transforms_mat_02v)
    transforms_mat = torch.matmul(transforms_mat, transforms_mat_02v_inv)
    transforms_mat[:, :3, 3] += transl.to(self.device)  # add global offset

    
    camera.data.update({
        'betas': betas, 
        'transforms_mat': transforms_mat,
        'rots': poses_all,
        'trans': transl.unsqueeze(0),
    })   