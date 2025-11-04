

# The code for Move-Exact and Move-Approx

This repo contains the source code for the Move-Exact and Move-Approx models.


## Programming language

Pytorch

## Required libraries

see requirements.txt

## Hardware info

GeForce GTX 1080 Ti 11 GB GPU

## Dataset info

The datasets include the check-in and check-out records of anonymous passengers from the Mass Transit Railway (MTR) Corporation, which is the only provider of subway services in Hong Kong for passengers, constituting 40% of the daily travel population[1].
We extract MTR data from 2020-11-01 to 2020-11-07, 2020-11-14, 2020-11-21, and 2020-11-28 as MTR-1 Week, MTR-2 Weeks, MTR-3 Weeks, and  MTR-4 Weeks datasets, respectively.

Since these datasets are confidential to MTR, we do not upload them in this repo. 

## Data preprocessing

We extract all check-in and check-out station and timestamp information for all passengers. For the trips that have missing check-in or check-out records, we filter out them in the evaluation. (The preprocessing code is at move_exact.py)



## How to run the code
```
python move_exact.py
python evaluation.py
python move_approx.py
```

## References
* [1] Sun W, Grubenmann T, Cheng R, et al. Modeling Long-Range Travelling Times with Big Railway Data[C]//International Conference on Database Systems for Advanced Applications, 2022: 443-454.
