#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
from scipy import spatial
import os
import cv2 as cv

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def load_camera_info(file_path):
    with open(file_path, "r") as fi:
        cam_file = fi.readlines()

    num_files = cam_file.pop(0)

    filenames = np.array([i.strip() for i in cam_file[0::7]])
    locations = np.array([np.fromstring(i, sep=" ") for i in cam_file[2::7]])

    filenames = filenames[:, None]

    locations = np.hstack([filenames, locations])

    rotation_matrices = np.array([np.fromstring(i, sep=" ") for i in cam_file[3::7]])

    translation_vectors = np.array([np.fromstring(i, sep=" ") for i in cam_file[4::7]])

    projection_matrices = np.array([np.fromstring(i, sep=" ") for i in cam_file[4::7]])

    cam_info = dict(filenames=filenames, locations=locations,
                    rotation_matrices=rotation_matrices,
                    translation_vectors=translation_vectors,
                    projection_matrices=projection_matrices)
    return cam_info


def get_stereo_pairs(_dir):
    cam_info_path = os.path.join(_dir, "camera_estimated.txt")
    cam_info = load_camera_info(cam_info_path)
    filenames = np.array([os.path.join(_dir, fi[0]) for fi in cam_info["filenames"]])

    translation_vectors = cam_info["translation_vectors"]

    angles = [[angle_between(i, j) for j in translation_vectors]
              for i in translation_vectors]
    angles = np.array(angles)

    deg5 = (5. / 365.) * 2 * np.pi
    deg45 = (45. / 365.) * 2 * np.pi

    angles[angles < deg5] = 10 ** 5
    angles[angles > deg45] = 10 ** 5

    distances = spatial.distance.pdist(translation_vectors)

    dist_mat = spatial.distance.squareform(distances)

    med_dist = np.median(dist_mat)

    dist_mat[dist_mat < 0.05 * med_dist] = 10 ** 5
    dist_mat[dist_mat > 2. * med_dist] = 10 ** 5

    final_mat = dist_mat * angles

    best_pairs = np.hstack([np.arange(len(filenames))[:, None], np.argmin(final_mat, axis=0)[:, None]])

    # Make sure all pairs are in L, R order
    for i in range(best_pairs.shape[0]):
        p1, p2 = best_pairs[i, :]
        if translation_vectors[p1][0] > translation_vectors[p2][0]:
            best_pairs[i] = np.array([p2, p1])

    best_pairs = filenames[best_pairs].reshape(best_pairs.shape)


def main():
    best_pairs = get_stereo_pairs("Monster/")
    disp_maps = []
    for pair in best_pairs:
        imgL = cv.pyrDown(cv.imread(pair[0]))  # downscale images for faster processing
        imgR = cv.pyrDown(cv.imread(pair[1]))

        # disparity range is tuned for 'aloe' image pair
        window_size = 3
        min_disp = 16
        num_disp = 112 - min_disp
        stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                      numDisparities=num_disp,
                                      blockSize=16,
                                      P1=8 * 3 * window_size ** 2,
                                      P2=32 * 3 * window_size ** 2,
                                      disp12MaxDiff=1,
                                      uniquenessRatio=10,
                                      speckleWindowSize=100,
                                      speckleRange=32
                                      )

        print('computing disparity...')
        disp_maps.append(stereo.compute(imgL, imgR).astype(np.float32) / 16.0)



    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
