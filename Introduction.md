# Set up for Stock GCP Project
## 1. Test Repo on GitHub
### 1.1 Create a new repository

* Fill in Repository name: gcp_test (DON'T use Capital Letter in repo name)

>Note: Use Capital Letter in GitHub Repo name will cause GCP build error: "Failed to trigger build: generic::invalid_argument: ..."  

* Choose Public
* Check Add a README file
* Check Add .gitignore
* Check Google Cloud Build
* Create repository

### 1.2 Clone the repo to local

* Homepage(<>Code) - Code (Clone) [[HTTPS or SSH: Connecting to GitHub with SSH](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/connecting-to-github-with-ssh)]
* Go to Git Bash or PowerShell
* Go to directory(~/workspace)
```
git clone <HTTPS URL or SSH URL>
```

## 2. Create a Conda Environment
### 2.1A Create conda environment
```
conda create -n test_gcp python==3.8
```
### 2.2 Create a environment.yml file

* Activate conda env
```
conda activate test_gcp
```
* Go to repo directory(~/workspace/gcp_test)
* Export environment file
```
//Command for Windows 
conda env export | findstr /v "^prefix: " > environment.yml 
//Command for Mac 
conda env export | grep -v "^prefix: " >environment.yml
```
>Note: grep is not available in windows, so we can use findstr instead.
>>Parameter /v  or -v means: Prints only lines that don't contain a match.
>>"^" means: Beginning line position - Beginning of the line.

### 2.1B Create conda environment by .yml file

```
//conda env create -f <.yml>
conda env create -f environment.yml
```
>Note: Need to change the encoding type to ANSI or UTF-8, **not GBK**
## 3. Commit and push to GitHub

* Modify the file .gitignore, under "MANIFEST" to add the ignore file rules
* Add and Commit modified files

```
git add <file> 
git commit -m "MESSAGE" 
//or add and commit with one command at the same time (It does not apply to adding new files) 
git commit -am "MESSAGE"
```

* Pull the latest changes on the GitHub (Recommend do Pull command before Push)
* Push the local modification to GitHub

```
git pull 
git push
```
## 4. Set up local Development Environment (PyCharm)

* Open the project
* Set up the configuration

    File - Settings (Ctrl+Alt+S)  
    -> Project: gcp_test(<Project Name>)  
    -> Python Interpretor  
        Automatically load the related conda environment(env_gcp_test)
        or Click on the wheel -> Add... -> **Conda Environment** -> Existing Environment
        Add Python files and run (The most efficient development mode: in PyCharm run app.py(main file), but need to complete cloud build, cloud run/deploy first to create the cloud bucket)
>Note: for running the program applying google bucket, need to **Authenticate** to Google Cloud API, cf. 5.2
If run the program in Powershell, need to activate the according conda environment

* Git commit (in PyCharm)

* Commit and Push(Ctrl+Alt+K)

## 5. Config Google Cloud Platform

* Go to the homepage of [Google Cloud Platform Console](http://console.cloud.google.com)

### 5.1 Add new project on GCP

* Navigation Menu - Home - Dashboard -> CREATE PROJECT (**CAN NOT** use "_" in project name, but support Capital Letter)

### 5.2 Authenticate to Google Cloud API

* Navigation Menu - IAM & Admin - Service Accounts -> + CREATE SERVICE ACCOUNT  

* Service account details:  

    * Service account name*: ->CREATE

* Grant this service account access to project:

    * Select a role: Project -  Owner ->CONTINUE ->DONE

* Select the Service Account -> ADD KEY -> Create new key - JSON -> CREATE

    * The .json file is downloaded automatically to local system.

>Note: You can add multiple Key files, but make sure that the KeyID exists in the Key list in the Service account details and the related .json file located in the specific path  
>
   * Save this file in a specific path, such as C:\Users\argus\GoogleServiceAccountKey\StockProject-34ad8e5311d6.json  

   * Add environment variable GOOGLE_APPLICATION_CREDENTIALS as the specific path in Windows System Environment  

>*Note: after adding environment variable, need to **reboot** the computer to apply the setting (How to use 2 .json file at the same time?)
### 5.3 Create build trigger(GitHub Repo to Cloud Build)

* Navigation Menu - Cloud Build - Dashboard - SET UP BUILD TRIGGER or Triggers - CREATE TRIGGERS

    * Name:
    * Event: Push to a branch
    * Source: CONNECT NEW REPOSITORY - Select your source - GitHub (Cloud Build GitHub App) - Continue - GitHub Account - Select repo ->Connect repository
    * ->Create push trigger

### 5.4 Enable Cloud Build API

* Navigation Menu - Cloud Build - Setting - View API -  Cloud Build API -> Enable
* Modify the source codes (in PyCharm)
* Git commit (in PyCharm)
* Commit and Push(Ctrl+Alt+K)

## 6. Deploy local Docker
### 6.1 Add a Dockerfile
```
FROM alpine 
CMD ["echo hello"]
```
>Note:  alpine is the common OS base
### 6.2 Build Docker
```
//docker build --tag <docker_image_name>(:<Version>) . (-f <Docker_file_name>) 
docker build -t docker_gcp_test:1.0 .
```
>Note:
>>-t or --tag: docker image tag
>>1.0 - Version Number, optional
>>. - current location
>>-f <Docker_file_name>: specific docker file, optional

* Check the docker images

```
docker images
```
### 6.3 Run Docker
```
//docker run <docker_image_name>:<Version> 
docker run docker_gcp_test:1.0 
docker run -p 8080:8080 -v GOOGLE_APPLICATION_CREDENTIALS:/creds.json -e GOOGLE_APPLICATION_CREDENTIALS=/creds.json docker_stock_test:2.0
```
>Note: run docker doesn't work with the feedback of Internal Server Error?
### 6.4 Debug Docker in interactive mode
```
docker run -it <docker_image_name>:<Version> /bin/sh 

docker run -it docker_gcp_test:1.0 /bin/sh 
docker run -it -p 8080:8080 -v GOOGLE_APPLICATION_CREDENTIALS:/creds.json -e GOOGLE_APPLICATION_CREDENTIALS=/creds.json docker_stock_test:2.0 /bin/bash
```
>Note: -it: interactive mode
>>Quit the interactive mode: Ctrl+D or exit
## 7. Deploy on GCP
### 7.1 Build Trigger by Git Push

* Git Push
* Trigger new Cloud Build automatically according to the **Dockerfile** (including create a **conda environment** by .yml fie) in the project directory

### 7.2 Check the build images in Container Registry

* Cloud Build - History - Choose One Build (Build Details, we can check BUILD LOG here) - BUILD ARTIFACTS - View Image
or
* Navigation Menu - Container Registry - Images

>We can **Activate Cloud Shell** by clicking ">_" button on the top-right of the page to run google cloud shell commands
### 7.3 Deploy to Cloud

* Images -> Digest details

* Deploy - Deploy to Cloud Run - Create Service

1.Service settings  
    Deployment platform: Cloud Run Region (Default: us-central1(lowa))
    Service name:
    Authentication: Allow unauthenticated invocations
>Note: Require authentication? Manage authorized users with Cloud IAM.
not sure about related to Google Service Account Key [GOOGLE_APPLICATION_CREDENTIALS]


2.Configure the service's first revision  
    Deploy one revision from an existing container image(Default)
    General: Maximum number of instances: 10 (restrain)
### 7.4 Cloud Run

* Navigation Menu - Cloud Run - Service details
* Copy URL to run (Automatically build will keep the same URL)

>Note: Sometimes, the Internal Server Error occurs. 
>>Delete current Cloud Run, and Deploy again (with the same service name, the URL remains unchanged)
>>Missing lib library, may cause Service Unavailable error



