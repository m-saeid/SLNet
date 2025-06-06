import numpy as np

class Translate_Pointcloud():
    def __init__(self):
        super().__init__()
    def transform(self, pcd):               # main AUG
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        
        translated_pcd = np.add(np.multiply(pcd, xyz1), xyz2) # .astype('float32')
        return translated_pcd


class Jitter():
    def __init__(self, sigma=0.01, clip=0.05):
        super().__init__()
        self.sigma = sigma
        self.clip = clip

    def transform(self, pcd):
        npts, nfeats = pcd.shape
        jit_pts = np.clip(self.sigma * np.random.randn(npts, nfeats), -self.clip, self.clip)
        jit_pts += pcd
        return jit_pts


class Rotation():
    def __init__(self, axis='y', angle=15):
        super().__init__()
        self.axis = axis
        self.angle = angle

    def transform(self, pcd):
        angle = np.random.uniform(-self.angle, self.angle)
        angle = np.pi * angle / 180
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        if self.axis == 'x':
            rotation_matrix = np.array([[1, 0, 0], [0, cos_theta, sin_theta], [0, -sin_theta, cos_theta]])
        elif self.axis == 'y':
            rotation_matrix = np.array([[cos_theta, 0, -sin_theta], [0, 1, 0], [sin_theta, 0, cos_theta]])
        elif self.axis == 'z':
            rotation_matrix = np.array([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]])
        else:
            raise ValueError(f'axis should be one of x, y and z, but got {self.axis}!')
        rotated_pts = pcd @ rotation_matrix
        return rotated_pts


class Translation():
    def __init__(self, shift=0.2):
        super().__init__()
        self.shift = shift

    def transform(self, pcd):
        npts = pcd.shape[0]
        x_translation = np.random.uniform(-self.shift, self.shift)
        y_translation = np.random.uniform(-self.shift, self.shift)
        z_translation = np.random.uniform(-self.shift, self.shift)
        x = np.full(npts, x_translation)
        y = np.full(npts, y_translation)
        z = np.full(npts, z_translation)
        translation = np.stack([x, y, z], axis=-1)
        translation_pts = pcd + translation
        return translation_pts


class AnisotropicScaling():
    def __init__(self, min_scale=0.66, max_scale=1.5):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def transform(self, pcd):
        x_factor = np.random.uniform(self.min_scale, self.max_scale)
        y_factor = np.random.uniform(self.min_scale, self.max_scale)
        z_factor = np.random.uniform(self.min_scale, self.max_scale)
        scale_matrix = np.array([[x_factor, 0, 0], [0, y_factor, 0], [0, 0, z_factor]])
        scaled_pts = pcd @ scale_matrix
        return scaled_pts
    

'''
class ShufflePointsOrder():
    def transform(self, results):
        idx = np.random.choice(results['pcd'].shape[0], results['pcd'].shape[0], replace=False)
        results['pcd'] = results['pcd'][idx]
        if 'seg_label' in results:
            results['seg_label'] = results['seg_label'][idx]
        return results
'''


class DataAugmentation():   # aug_type=[none, random, translate_pointcloud, jitter, rotation, translation, anisotropic_scaling]
    def __init__(self, aug_type='random', axis='y', angle=15, shift=0.2, min_scale=0.66, max_scale=1.5, sigma=0.01, clip=0.05):
        super().__init__()
        self.aug_type = aug_type
        jitter = Jitter(sigma, clip)
        rotation = Rotation(axis, angle)
        translation = Translation(shift)
        anisotropic_scaling = AnisotropicScaling(min_scale, max_scale)
        translate_pointcloud = Translate_Pointcloud()
        self.aug_list = [translate_pointcloud, jitter, rotation, translation, anisotropic_scaling]

    def transform(self, pcd):
        if self.aug_type=='no':
            return pcd
        elif self.aug_type=='random':
            pcd = np.random.choice(self.aug_list).transform(pcd)
        elif self.aug_type=='translate_pointcloud':
            pcd = self.aug_list[0].transform(pcd)
        elif self.aug_type=='jitter':
            pcd = self.aug_list[1].transform(pcd)
        elif self.aug_type=='rotation':
            pcd = self.aug_list[2].transform(pcd)
        elif self.aug_type=='translation':
            pcd = self.aug_list[3].transform(pcd)
        elif self.aug_type=='anisotropic_scaling':
            pcd = self.aug_list[4].transform(pcd)
        else:
            raise Exception(f"aug_type: {self.aug_type}")
        return pcd.astype('float32')
    

if __name__ == '__main__':
    pcd = np.random.rand(1024,3)
    for i in range(10):
        aug = DataAugmentation(aug_type='jitter')
        out = aug.transform(pcd)
        print(out.shape)
        print(out)
