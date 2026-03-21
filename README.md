Application Setup
	Using Docker

	Install the Docker extension in Visual Studio Code.
	
	Ensure Docker is installed and running.
	
	Connect Visual Studio Code to WSL.
	
	Reopen the project inside the Dev Container.

	Training the Model

		Run the dataset download script to retrieve the dataset from Kaggle:
		
			python dataset_download.py -url "PASTE KAGGLE DATASET URL" 

		Start training the model:

			/workspaces/generative_model/train.py --img_size 64 --ds_folder_name animefacedataset --latent_dim 128 --dis_lr 0.0001 --gen_lr 0.0003 --epochs 20 --batch_size 512

	Generating Images

		Run the generation script:
			
			/workspaces/generative_model/generate.py --img_size 64 --img_name sample_1 --latent_dim 128 --ds_name animefacedataset --gen_img_batch 256 

Important:
- The models/ directory must contain the trained .pth model files (generator).
- If these files are missing, the generator will not be able to produce images.

Be aware that the latent_used for training should be the same when generating as the autoencoder is trained with that specific latent_dim