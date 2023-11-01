# Variational Model Inversion Attacks
Kuan-Chieh Wang, Yan Fu, Ke Li, Ashish Khisti, Richard Zemel, Alireza Makhzani

![Fig1](./figs/fig1.png)
* Most commands are in `run_scripts`.  
* We outline a few example commands here.  
	* Commands below end with a suffix `<mode>`.  Setting `<mode>=0` will run code locally.  `<mode>=1` was used with SLURM on a computing cluster. 
* The environment variable `ROOT1` was set to my home directory. 

## Set up task (data & pretrained models, etc.)
Check out the StyleGAN [repo](https://github.com/wangkua1/stylegan2-ada-pytorch) and place it in the same directory hierarchy as the present repo.  This is used to make sure you can load and run the pretrained StyleGAN checkpoints.

For CelebA experiments:  
* Data -- 
	* download the "Align&Cropped Images" from the [CelebA website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) into the directory `data/img_align_celeba`. 
	* make sure in `data/img_align_celeba`, there are 000001.jpg to 202599.jpg.
	* download `identity_CelebA.txt` and put it in `data/celeb_a`.
* Pretrained DCGAN -- download and untar [this](https://drive.google.com/file/d/1omxX6bg2YI-kvMIQK-eOj7PK1m04cMir/view?usp=sharing) into the folder `pretrained/gans/neurips2021-celeba`.
* Pretrained StyleGAN -- download and untar [this](https://drive.google.com/file/d/1jS0YFurFb56pfHj_Iw7LBRFj_mGymTen/view?usp=sharing) into the folder `pretrained/stylegan/neurips2021-celeba`.
* Pretrained Target Classifier -- download and untar [this](https://drive.google.com/file/d/1pTw1CZsXK5auEntL5NE490Jsa69dA7cv/view?usp=sharing) into the folder `pretrained/classifiers/neurips2021-celeba`.
* Evaluation Classifier --
	* check out the InsightFace [repo](https://github.com/wangkua1/InsightFace_Pytorch) and place it in the same directory hierarchy as the present repo. 
	* follow instructions in that repo, and download the `ir_se50` model, which is used as the evaluation classifier. 


## Train VMI 
**CelebA**
* the script below runs VMI attack on the first 100 IDs and saves the results to `results/celeba-id<ID>`.
```
run_scripts/neurips2021-celeba-stylegan-flow.sh
```
* generate and aggregate the attack samples by running the command below. The results will be saved to `results/images_pt/stylegan-attack-with-labels-id0-100.pt`.
```
python generate_vmi_attack_samples.py
```
* evaluate the generated samples by running:
```
fprefix=results/images_pt/stylegan-attack-with-labels-id0-100

python evaluate_samples.py \
	--name load_samples_pt \
	--samples_pt_prefix $fprefix \
	--eval_what stats \
	--nclass 100
```



# Acknowledgements
Code contain snippets from:   
https://github.com/adjidieng/PresGANs  
https://github.com/pytorch/examples/tree/master/mnist   
https://github.com/wyharveychen/CloserLookFewShot

