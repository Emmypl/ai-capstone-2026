# ai-capstone-hw4
NYCU AI Capstone 2026 Spring

In your original ai-capstone directory, `git pull` to get new `hw4` directory.

# hw4 Robot Manipulation

Homework4 Document: [Link](https://docs.google.com/document/d/1K29VWRyrtk-1Y8lDYBHqhRm5T5tt8XI2F1sLEA7vJ9w/edit?usp=sharing) 

## Installation


There are two ways to create your environment(choose one for your hw4):

first way(on Glows.ai):
1. [Glows.ai](https://docs.google.com/presentation/d/1oKdlhMkB-DIS8_hGv_97MqcMtsDmft48aCoy6GeyW1A/edit?usp=sharing)

second way(on Local machine):

choose one way to develope your homework 4
1. Create pip environment(Ubuntu)
```shell
cd hw4
# 安裝全部
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync

# Task 1 with GUI
python fk.py
# Task 2 with GUI
python ik.py
```
2. Using docker environment, then install Python packages.(Ubuntu/MacOS)
```shell
cd hw4
# Task 1 without GUI
python3 docker.py run-fk --headless

# Task 2 with GUI
python3 docker.py run-ik

#or

# Task 2 without GUI
python3 docker.py run-ik --headless

# Task 2 with GUI
python3 docker.py run-ik
```
## Task 1

Implement your_fk function in fk.py
- execution example
```bash
python fk.py
pybullet build time: Sep 22 2020 00:55:20
============================ Task 1 : Forward Kinematic ============================

- Testcase file : fk_test_case_ta1.json
- Your Score Of Forward Kinematic : 5.000 / 5.000, Error Count :    0 /  100
- Your Score Of Jacobian Matrix   : 5.000 / 5.000, Error Count :    0 /  100

- Testcase file : fk_test_case_ta2.json
- Your Score Of Forward Kinematic : 5.000 / 5.000, Error Count :    0 /  100
- Your Score Of Jacobian Matrix   : 5.000 / 5.000, Error Count :    0 /  100

====================================================================================
- Your Total Score : 20.000 / 20.000
====================================================================================
```

## Task 2

Implement your_ik function in ik.py
- execution example
```bash
python ik.py
###################################### log #########################################

============================ Task 2 : Inverse Kinematic ============================

- Testcase file : ik_test_case_ta1.json
- Mean Error : 0.001048
- Error Count :   0 / 100
- Your Score Of Inverse Kinematic : 20.000 / 20.000

- Testcase file : ik_test_case_ta2.json
- Mean Error : 0.001482
- Error Count :   0 / 100
- Your Score Of Inverse Kinematic : 20.000 / 20.000

====================================================================================
- Your Total Score : 40.000 / 40.000
====================================================================================
```
