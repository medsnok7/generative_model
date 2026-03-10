Application Setup
	Using Docker

	Install the Docker extension in Visual Studio Code.
	
	Ensure Docker is installed and running.
	
	Connect Visual Studio Code to WSL.
	
	Reopen the project inside the Dev Container.

	Training the Model

		Run the dataset download script to retrieve the dataset from Kaggle:
		
			python datasetdownload.py

		Start training the model:

			python train.py --latent_dim 128 --ds_folder_name animefacedataset --is_cmplx 0 --dis_lr 0.0001 --gen_lr 0.0003 --epochs 60

	Generating Images

		Run the generation script:

			python generate.py --img_name jellyfish --ds_folder_name animefacedatase --is_cmplx 0 --latent_dim 128

Important:
- The models/ directory must contain the trained .pth model files (generator and discriminator weights).
- If these files are missing, the generator will not be able to produce images.


When using the example of trained models, consider that:
- The animefacedataset model is trained with latent_dim 128 and 64x64 model
	->  python generate.py --img_name jellyfish --ds_folder_name animefacedataset --is_cmplx 0 --latent_dim 128 
- Jellyfish-types model is trained with latent_dim 128 and 128x128 model 
  	->  python generate.py --img_name jellyfish --ds_folder_name jellyfish-types --is_cmplx 1 --latent_dim 128 

Be aware that the latent_used for training should be the same when generating as the autoencoder is trained with that specific latent_dim