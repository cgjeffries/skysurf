import numpy as np

from skysurf_estimate_sky import calculate_sky
from make_diagnostic import make_plots

# For downloading test data
from astropy.io import fits
#from astroquery.mast import Observations

# Other
from glob import glob
import pandas as pd
import os

from multiprocessing import Pool


### Query observations ###
#obs_table = Observations.query_criteria(proposal_pi="Jansen*", proposal_id = 15278, filters = 'F606W')
#data_products = Observations.get_product_list(obs_table)

### Download FLC files ###
# data_flc = data_products[(data_products['productSubGroupDescription'] == 'FLC') & (data_products['type'] == 'S')]
# Observations.download_products(data_flc[:10]) # Only download the first few images

#file_list = glob('mastDownload/HST/*/*_flc.fits')
# file_list = glob('../../my_stuff/bad_images/*/*_flc.fits')
#
#
# file = file_list[0]

#
# # Open the fits file
# hdu = fits.open(file)
#
# # Open the science data
# sci1_data = hdu['SCI', 1].data
#
# # Open the data quality data
# dq1_data = hdu['DQ', 1].data
#
#
# sky_dic1 = calculate_sky(sci1_data, bin_size = 64, dq_data = dq1_data, has_DQ = True, dq_fraction = 0.2,
#                          percentile = 50)
#
#
# make_plots(data = sci1_data, cutouts = sky_dic1['cutouts'][0],
#            goodind = sky_dic1['lowest5perc_ind'][0], badind = sky_dic1['bad_ind'][0],
#            sky = sky_dic1['calc_sky'][0], rms = sky_dic1['calc_rms'][0],
#            badpx = sky_dic1['badpx_ind'][0], title = file_list[0],
#            save = False, savepath = None,
#            show = True, figsize = (15,9))
#
# df = pd.DataFrame([])

def myWorker(file):
    for sci_ext in [1, 2]:

        print(file, 'SCI' + str(sci_ext))

        # Make small dataframe with file name
        root = os.path.basename(file).split('_')[0]
        file_df = pd.DataFrame({'file': [file], 'root': [root], 'sci_ext': [sci_ext]})

        # Open the fits file
        hdu = fits.open(file)

        # Save the science data
        sci_data = hdu['SCI', sci_ext].data

        # Save the data quality data
        dq_data = hdu['DQ', sci_ext].data

        #sky_dic = calculate_sky(sci_data, bin_size=64, dq_data=dq_data, has_DQ=True, dq_fraction=0.2)
        sky_dic = calculate_sky(sci_data, bin_size=64, dq_data=dq_data, has_DQ=False, dq_fraction=0.2)

        ### Make plots ###
        # Instead of showing each figure, save them in the local directory
        save_images = '{r}_percentileclip_sky_SCI{e}.png'.format(r=root, e=sci_ext)
        title = file_list[0] + '\nExt. {}'.format(sci_ext)
        if not os.path.exists(f"outputs/{root}/{sci_ext}"):
            os.makedirs(f"outputs/{root}/{sci_ext}")
        np.save(f"outputs/{root}/{sci_ext}/SCI.npy", np.flipud(sci_data))

        mask_shape = (int(sci_data.shape[1]/64), int(sci_data.shape[0]/64))
        row_indices, col_indices = np.unravel_index(sky_dic['bad_ind'][0], mask_shape)
        mask = np.zeros(mask_shape)
        mask[row_indices, col_indices] = 1
        #mask = np.flip(np.transpose(mask), axis=1)
        mask = np.rot90(mask, k=1)
        np.save(f"outputs/{root}/{sci_ext}/MASK.npy", mask)
        # make_plots(data=sci_data, cutouts=sky_dic['cutouts'][0],
        #            goodind=sky_dic['lowest5perc_ind'][0], badind=sky_dic['bad_ind'][0],
        #            sky=sky_dic['calc_sky'][0], rms=sky_dic['calc_rms'][0],
        #            badpx=sky_dic['badpx_ind'][0], title=title,
        #            save=True, savepath=save_images,
        #            show=False, figsize=(15, 9))

        ### Drop keys from output that we no longer need ###
        drop_keys = ['sky_arr', 'rms_arr', 'cutouts', 'lowest5perc_ind', 'bad_ind', 'badpx_ind']
        for key in drop_keys:
            sky_dic.pop(key, None)

        # ### Save information to dataframe ###
        # tempdf = pd.DataFrame(pd.concat([file_df, pd.DataFrame(sky_dic)], axis=1))
        #
        # df = df.append(tempdf)



#for file in file_list:

if __name__ == "__main__":  # confirms that the code is under main function
    file_list = glob('../../my_stuff/bad_images/*/*_flc.fits')
    print(f"processing {len(file_list)} images...")
    p = Pool(20)
    p.map(myWorker, file_list)



#df.to_csv('percentileclip_sky_jwstneptdf.csv', index = False)