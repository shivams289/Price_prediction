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

#File Structure
- Temp_repo
  1. data
      - mercari_train.csv
      - mercari_tes.csv
  2. model
      - model.pkl
  3. model.py
  4. build_model.py
  5. submission.csv
  6. environment.yml
  7. sample_submission.csv
  8. app.py
  9. readme.md
  10. SKILL_TEST.en.md
