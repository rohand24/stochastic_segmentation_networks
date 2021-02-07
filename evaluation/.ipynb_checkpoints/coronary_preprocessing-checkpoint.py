import os
import SimpleITK as sitk
import argparse
import pandas as pd
import numpy as np
import glob

train_ids = ['CA_001',
             'CA_002',
             'CA_004',
             'CA_006',
             'CA_008',
             'CA_009',
             'CA_012', 
             'CA_013',
             'CA_014', 
             'CA_016',
             'CA_017',
             'CA_018',
             'CA_019',
             'CA_020',
             'CA_021',
             'CA_024',
             'CA_025',
             'CA_026',
             'CA_027',
             'CA_028',
             'CA_029',
             'CA_030',
             'CA_032',
             'CA_033',
             'CA_035',
             'CA_036',
             'CA_038',
             'CA_039',
             'CA_040',
             'CA_041',
             'CA_042',
             'CA_045',
             'CA_047',
             'CA_048',
             'CA_050',
             'CA_051',
             'CA_053',
             'CA_055',
             'CA_056',
             'CA_057',
             'CA_058',
             'CA_060',
             'CA_061',
             'CA_062',
             'CA_063',
             'CA_064',
             'CA_065',
             'CA_066',
             'CA_067', 
             'CA_070',
             'CA_071', 
             'CA_072',
             'CA_074',
             'CA_075',
             'CA_078',
             'CA_079',
             'CA_080',
             'CA_081',
             'CA_082',
             'CA_083', 
             'CA_084',
             'CA_085',
             'CA_086',
             'CA_088', 
             'CA_090',
             'CA_091',
             'CA_092',
             'CA_094',
             'CA_096',
             'CA_097',
             'CA_098',
             'CA_099',
             'CA_100']

valid_ids = ['CA_003',
             'CA_005',
             'CA_007',
             'CA_011',
             'CA_015',
             'CA_023',
             'CA_034', 
             'CA_037',
             'CA_043', 
             'CA_044',
             'CA_046',
             'CA_049', 
             'CA_052',
             'CA_059',
             'CA_069',
             'CA_073',
             'CA_077',
             'CA_089',
             'CA_093']

# test_ids = ['HGG/Brats17_2013_10_1/Brats17_2013_10_1',
#             'HGG/Brats17_2013_12_1/Brats17_2013_12_1',
#             'HGG/Brats17_2013_14_1/Brats17_2013_14_1',
#             'HGG/Brats17_2013_18_1/Brats17_2013_18_1',
#             'HGG/Brats17_2013_20_1/Brats17_2013_20_1',
#             'HGG/Brats17_2013_27_1/Brats17_2013_27_1',
#             'HGG/Brats17_2013_3_1/Brats17_2013_3_1',
#             'HGG/Brats17_2013_5_1/Brats17_2013_5_1',
#             'HGG/Brats17_CBICA_AAB_1/Brats17_CBICA_AAB_1',
#             'HGG/Brats17_CBICA_ABB_1/Brats17_CBICA_ABB_1',
#             'HGG/Brats17_CBICA_ABO_1/Brats17_CBICA_ABO_1',
#             'HGG/Brats17_CBICA_ALN_1/Brats17_CBICA_ALN_1',
#             'HGG/Brats17_CBICA_ANI_1/Brats17_CBICA_ANI_1',
#             'HGG/Brats17_CBICA_ANZ_1/Brats17_CBICA_ANZ_1',
#             'HGG/Brats17_CBICA_AOH_1/Brats17_CBICA_AOH_1',
#             'HGG/Brats17_CBICA_AOP_1/Brats17_CBICA_AOP_1',
#             'HGG/Brats17_CBICA_APR_1/Brats17_CBICA_APR_1',
#             'HGG/Brats17_CBICA_APZ_1/Brats17_CBICA_APZ_1',
#             'HGG/Brats17_CBICA_AQD_1/Brats17_CBICA_AQD_1',
#             'HGG/Brats17_CBICA_AQJ_1/Brats17_CBICA_AQJ_1',
#             'HGG/Brats17_CBICA_AQO_1/Brats17_CBICA_AQO_1',
#             'HGG/Brats17_CBICA_AQQ_1/Brats17_CBICA_AQQ_1',
#             'HGG/Brats17_CBICA_AQT_1/Brats17_CBICA_AQT_1',
#             'HGG/Brats17_CBICA_AQZ_1/Brats17_CBICA_AQZ_1',
#             'HGG/Brats17_CBICA_ASG_1/Brats17_CBICA_ASG_1',
#             'HGG/Brats17_CBICA_ASO_1/Brats17_CBICA_ASO_1',
#             'HGG/Brats17_CBICA_ATD_1/Brats17_CBICA_ATD_1',
#             'HGG/Brats17_CBICA_AVV_1/Brats17_CBICA_AVV_1',
#             'HGG/Brats17_CBICA_AWH_1/Brats17_CBICA_AWH_1',
#             'HGG/Brats17_CBICA_AXM_1/Brats17_CBICA_AXM_1',
#             'HGG/Brats17_CBICA_AXO_1/Brats17_CBICA_AXO_1',
#             'HGG/Brats17_CBICA_AYW_1/Brats17_CBICA_AYW_1',
#             'HGG/Brats17_CBICA_BHK_1/Brats17_CBICA_BHK_1',
#             'HGG/Brats17_TCIA_105_1/Brats17_TCIA_105_1',
#             'HGG/Brats17_TCIA_113_1/Brats17_TCIA_113_1',
#             'HGG/Brats17_TCIA_118_1/Brats17_TCIA_118_1',
#             'HGG/Brats17_TCIA_131_1/Brats17_TCIA_131_1',
#             'HGG/Brats17_TCIA_135_1/Brats17_TCIA_135_1',
#             'HGG/Brats17_TCIA_147_1/Brats17_TCIA_147_1',
#             'HGG/Brats17_TCIA_162_1/Brats17_TCIA_162_1',
#             'HGG/Brats17_TCIA_171_1/Brats17_TCIA_171_1',
#             'HGG/Brats17_TCIA_180_1/Brats17_TCIA_180_1',
#             'HGG/Brats17_TCIA_208_1/Brats17_TCIA_208_1',
#             'HGG/Brats17_TCIA_235_1/Brats17_TCIA_235_1',
#             'HGG/Brats17_TCIA_265_1/Brats17_TCIA_265_1',
#             'HGG/Brats17_TCIA_290_1/Brats17_TCIA_290_1',
#             'HGG/Brats17_TCIA_328_1/Brats17_TCIA_328_1',
#             'HGG/Brats17_TCIA_332_1/Brats17_TCIA_332_1',
#             'HGG/Brats17_TCIA_338_1/Brats17_TCIA_338_1',
#             'HGG/Brats17_TCIA_361_1/Brats17_TCIA_361_1',
#             'HGG/Brats17_TCIA_370_1/Brats17_TCIA_370_1',
#             'HGG/Brats17_TCIA_374_1/Brats17_TCIA_374_1',
#             'HGG/Brats17_TCIA_377_1/Brats17_TCIA_377_1',
#             'HGG/Brats17_TCIA_396_1/Brats17_TCIA_396_1',
#             'HGG/Brats17_TCIA_411_1/Brats17_TCIA_411_1',
#             'HGG/Brats17_TCIA_419_1/Brats17_TCIA_419_1',
#             'HGG/Brats17_TCIA_429_1/Brats17_TCIA_429_1',
#             'HGG/Brats17_TCIA_473_1/Brats17_TCIA_473_1',
#             'HGG/Brats17_TCIA_478_1/Brats17_TCIA_478_1',
#             'HGG/Brats17_TCIA_491_1/Brats17_TCIA_491_1',
#             'HGG/Brats17_TCIA_605_1/Brats17_TCIA_605_1',
#             'LGG/Brats17_2013_0_1/Brats17_2013_0_1',
#             'LGG/Brats17_2013_16_1/Brats17_2013_16_1',
#             'LGG/Brats17_2013_29_1/Brats17_2013_29_1',
#             'LGG/Brats17_2013_8_1/Brats17_2013_8_1',
#             'LGG/Brats17_TCIA_101_1/Brats17_TCIA_101_1',
#             'LGG/Brats17_TCIA_109_1/Brats17_TCIA_109_1',
#             'LGG/Brats17_TCIA_141_1/Brats17_TCIA_141_1',
#             'LGG/Brats17_TCIA_175_1/Brats17_TCIA_175_1',
#             'LGG/Brats17_TCIA_255_1/Brats17_TCIA_255_1',
#             'LGG/Brats17_TCIA_266_1/Brats17_TCIA_266_1',
#             'LGG/Brats17_TCIA_299_1/Brats17_TCIA_299_1',
#             'LGG/Brats17_TCIA_310_1/Brats17_TCIA_310_1',
#             'LGG/Brats17_TCIA_387_1/Brats17_TCIA_387_1',
#             'LGG/Brats17_TCIA_402_1/Brats17_TCIA_402_1',
#             'LGG/Brats17_TCIA_410_1/Brats17_TCIA_410_1',
#             'LGG/Brats17_TCIA_420_1/Brats17_TCIA_420_1',
#             'LGG/Brats17_TCIA_442_1/Brats17_TCIA_442_1',
#             'LGG/Brats17_TCIA_451_1/Brats17_TCIA_451_1',
#             'LGG/Brats17_TCIA_466_1/Brats17_TCIA_466_1',
#             'LGG/Brats17_TCIA_493_1/Brats17_TCIA_493_1',
#             'LGG/Brats17_TCIA_618_1/Brats17_TCIA_618_1',
#             'LGG/Brats17_TCIA_630_1/Brats17_TCIA_630_1',
#             'LGG/Brats17_TCIA_633_1/Brats17_TCIA_633_1',
#             'LGG/Brats17_TCIA_644_1/Brats17_TCIA_644_1',
#             'LGG/Brats17_TCIA_650_1/Brats17_TCIA_650_1']


def get_brain_mask(t1):
    brain_mask = sitk.GetImageFromArray((sitk.GetArrayFromImage(t1) > 0).astype(np.uint8))
    brain_mask.CopyInformation(t1)
    brain_mask = sitk.Cast(brain_mask, sitk.sitkUInt8)
    return brain_mask


def z_score_normalisation(channel, brain_mask, cutoff_percentiles=(5., 95.), cutoff_below_mean=True):
    low, high = np.percentile(channel[brain_mask.astype(np.bool)], cutoff_percentiles)
    norm_mask = np.logical_and(brain_mask, np.logical_and(channel > low, channel < high))
    if cutoff_below_mean:
        norm_mask = np.logical_and(norm_mask, channel > np.mean(channel))
    masked_channel = channel[norm_mask]
    normalised_channel = (channel - np.mean(masked_channel)) / np.std(masked_channel)
    return normalised_channel


def fix_segmentation_labels(seg):
    array = sitk.GetArrayFromImage(seg)
    array[array == 4] = 3
    new_seg = sitk.GetImageFromArray(array)
    new_seg.CopyInformation(seg)
    return new_seg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',
                        required=True,
                        type=str,
                        help='Path to input directory.')
    parser.add_argument('--output-dir',
                        required=True,
                        type=str,
                        help='Path to output directory.')

    parse_args, unknown = parser.parse_known_args()
    output_dataframe = pd.DataFrame()
    input_dir = parse_args.input_dir
    output_dir = parse_args.output_dir
    
    all_files = glob.glob(input_dir + '/labelsTr/*.nii.gz')
    img_path = input_dir + '/imagesTr/'
#     for subdir_1 in os.listdir(os.path.join(input_dir)):
    for seg_file in all_files:
        id_ = seg_file.split('/')[-1].split('.')[0]
        print(id_)
        seg = sitk.ReadImage(seg_file)#fix_segmentation_labels(sitk.ReadImage(os.path.join(parse_args.input_dir, id_) + '_seg.nii.gz'))
        seg_output_path = os.path.join(parse_args.output_dir, 'seg',id_) + f'.nii.gz'
        output_dataframe.loc[id_, 'seg'] = seg_output_path
        os.makedirs(os.path.dirname(seg_output_path), exist_ok=True)
        sitk.WriteImage(seg, seg_output_path)
        
        img_file = img_path+id_+'_0000.nii.gz'
        img = sitk.ReadImage(img_file)
        img_output_path = os.path.join(parse_args.output_dir, 'img',id_) + f'.nii.gz'
        output_dataframe.loc[id_, 'img'] = img_output_path
        os.makedirs(os.path.dirname(img_output_path), exist_ok=True)
        sitk.WriteImage(img, img_output_path)

        
    output_dataframe.index.name = 'id'
    os.makedirs('assets/Coronary_data', exist_ok=True)
    
    train_index = output_dataframe.loc[train_ids]
    train_index.to_csv('assets/Coronary_data/data_index_train.csv')
    valid_index = output_dataframe.loc[valid_ids]
    valid_index.to_csv('assets/Coronary_data/data_index_valid.csv')
#     test_index = output_dataframe.loc[test_ids]
#     test_index.to_csv('assets/Coronary_data/data_index_test.csv')
