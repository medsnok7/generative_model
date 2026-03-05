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

			python train.py

	Generating Images

		Run the generation script:

			python generate.py

Important:
The models/ directory must contain the trained .pth model files (generator and discriminator weights).
If these files are missing, the generator will not be able to produce images.
