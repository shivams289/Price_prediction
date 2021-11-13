# Temp_repo
- clone this repo using `git clone <https link>`
- change directory `cd Temp_repo`
- Create your virtual environment `conda env create --file environment.yml`
- activate this conda environment using `cond activate mercari`
- create another branch `git checkout -b new_branch` to make any changes leaving master alone
- check current branch using `git branch`
- `git checkout master` to go back to master branch
- run `build_model.py` to run the model and check if it runs sucessfully,  it will save it in model.pkl file and create submission.csv
- `app.py` needs modeification to create the api
