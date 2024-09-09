# Section Thermal Fabric

# Table B16.6-1. Total Furnace Load (GJ)
| Case                       | ESP-r/HOT3000 | EnergyPlus | DOE-2.1E |     |   Min |   Max |  Mean | Dev % $$ |     | Analytical/Quasi-Analytical |  TEST | 
|:-------------------------- | -------------:| ----------:| --------:| ---:| -----:| -----:| -----:| --------:| ---:| ---------------------------:| -----:| 
| HE100 100% eff.            |         77.94 |      77.75 |    77.75 |     | 77.75 | 77.94 |       |      0.2 |     |                       77.74 | 77.75 | 
| HE110 80% eff.             |         77.94 |      77.75 |    77.75 |     | 77.75 | 77.94 |       |      0.2 |     |                       77.74 | 77.75 | 
| HE120 80% eff., PLR=0.4    |         31.25 |      31.10 |    31.13 |     | 31.10 | 31.25 |       |      0.5 |     |                       31.10 | 31.13 | 
| HE130 No Load              |          0.00 |       0.00 |     0.15 |     |  0.00 |  0.15 |       |        - |     |                        0.00 |  0.15 | 
| HE140 Periodic PLR         |         31.26 |      31.10 |    31.12 |     | 31.10 | 31.26 |       |      0.5 |     |                       31.10 | 31.12 | 
| HE150 Continuous Circ. Fan |         29.88 |      29.59 |    29.57 |     | 29.57 | 29.88 |       |      1.1 |     |                       29.65 | 29.57 | 
| HE160 Cycling Circ. Fan    |         31.26 |      30.46 |    30.49 |     | 30.46 | 31.26 |       |      2.6 |     |                       31.10 | 30.49 | 
| HE170 Draft Fan            |         29.88 |      29.59 |    29.57 |     | 29.57 | 29.88 |       |      1.1 |     |                       29.65 | 29.57 | 
|                            | 
| HE210 Realistic Weather    |         41.36 |      42.04 |    42.06 |     | 41.36 | 42.06 | 41.82 |      1.7 |     |                             | 42.06 | 
| HE220 Setback Thermostat   |         39.41 |      39.87 |    39.76 |     | 39.41 | 39.87 | 39.68 |      1.2 |     |                             | 39.76 | 
| HE230 Undersized Furnace   |         34.32 |      34.59 |    34.37 |     | 34.32 | 34.59 | 34.43 |      0.8 |     |                             | 34.37 | 

$$ For HE1xx cases ABS[ (Max-Min) / (Analytic Solution)] and for HE2xx cases ABS[ (Max-Min) / (Mean of Example Simulation Results)]


# Table B16.6-2. Total Furnace Input (GJ)
| Case                       | ESP-r/HOT3000 | EnergyPlus | DOE-2.1E |     |   Min |   Max |  Mean | Dev % $$ |     | Analytical/Quasi-Analytical |  TEST | 
|:-------------------------- | -------------:| ----------:| --------:| ---:| -----:| -----:| -----:| --------:| ---:| ---------------------------:| -----:| 
| HE100 100% eff.            |         77.74 |      77.71 |    78.42 |     | 77.71 | 78.42 |       |      0.9 |     |                       77.71 | 78.42 | 
| HE110 80% eff.             |         96.92 |      97.22 |    98.02 |     | 96.92 | 98.02 |       |      1.1 |     |                       97.22 | 98.02 | 
| HE120 80% eff., PLR=0.4    |         38.41 |      38.27 |    38.56 |     | 38.27 | 38.56 |       |      0.8 |     |                       38.27 | 38.56 | 
| HE130 No Load              |          0.00 |       0.00 |     0.14 |     |  0.00 |  0.14 |       |        - |     |                        0.00 |  0.14 | 
| HE140 Periodic PLR         |         39.00 |      39.00 |    38.76 |     | 38.76 | 39.00 |       |      0.6 |     |                       39.00 | 38.76 | 
| HE150 Continuous Circ. Fan |         37.23 |      36.94 |    36.82 |     | 36.82 | 37.23 |       |      1.1 |     |                       37.02 | 36.82 | 
| HE160 Cycling Circ. Fan    |         38.12 |      38.12 |    37.96 |     | 37.96 | 38.12 |       |      0.4 |     |                       38.09 | 37.96 | 
| HE170 Draft Fan            |         37.23 |      36.94 |    36.82 |     | 36.82 | 37.23 |       |      1.1 |     |                       37.02 | 36.82 | 
|                            | 
| HE210 Realistic Weather    |         50.53 |      52.01 |    52.37 |     | 50.53 | 52.37 | 51.64 |      3.6 |     |                             | 52.37 | 
| HE220 Setback Thermostat   |         47.87 |      49.35 |    49.47 |     | 47.87 | 49.47 | 48.89 |      3.3 |     |                             | 49.47 | 
| HE230 Undersized Furnace   |         41.37 |      42.55 |    43.22 |     | 41.37 | 43.22 | 42.38 |      4.4 |     |                             | 43.22 | 

$$ For HE1xx cases ABS[ (Max-Min) / (Analytic Solution)] and for HE2xx cases ABS[ (Max-Min) / (Mean of Example Simulation Results)]


# Table B16.6-3. Fuel Consumption (m3/2)
| Case                       | ESP-r/HOT3000 | EnergyPlus | DOE-2.1E |     |      Min |      Max |     Mean | Dev % $$ |     | Analytical/Quasi-Analytical |     TEST | 
|:-------------------------- | -------------:| ----------:| --------:| ---:| --------:| --------:| --------:| --------:| ---:| ---------------------------:| --------:| 
| HE100 100% eff.            |      0.000263 |   0.000263 | 0.000265 |     | 0.000263 | 0.000265 |          |      0.9 |     |                    0.000263 | 0.000265 | 
| HE110 80% eff.             |      0.000328 |   0.000329 | 0.000332 |     | 0.000328 | 0.000332 |          |      1.1 |     |                    0.000329 | 0.000332 | 
| HE120 80% eff., PLR=0.4    |      0.000130 |   0.000130 | 0.000131 |     | 0.000130 | 0.000131 |          |      0.8 |     |                    0.000130 | 0.000131 | 
| HE130 No Load              |      0.000000 |   0.000000 | 0.000000 |     | 0.000000 | 0.000000 |          |        - |     |                    0.000000 | 0.000000 | 
| HE140 Periodic PLR         |      0.000132 |   0.000132 | 0.000131 |     | 0.000131 | 0.000132 |          |      0.6 |     |                    0.000132 | 0.000131 | 
| HE150 Continuous Circ. Fan |      0.000126 |   0.000125 | 0.000125 |     | 0.000125 | 0.000126 |          |      1.1 |     |                    0.000125 | 0.000125 | 
| HE160 Cycling Circ. Fan    |      0.000129 |   0.000129 | 0.000129 |     | 0.000129 | 0.000129 |          |      0.4 |     |                    0.000129 | 0.000129 | 
| HE170 Draft Fan            |      0.000126 |   0.000125 | 0.000125 |     | 0.000125 | 0.000126 |          |      1.1 |     |                    0.000125 | 0.000125 | 
|                            | 
| HE210 Realistic Weather    |      0.000171 |   0.000176 | 0.000177 |     | 0.000171 | 0.000177 | 0.000175 |      3.5 |     |                             | 0.000177 | 
| HE220 Setback Thermostat   |      0.000162 |   0.000167 | 0.000167 |     | 0.000162 | 0.000167 | 0.000165 |      3.3 |     |                             | 0.000167 | 
| HE230 Undersized Furnace   |      0.000140 |   0.000144 | 0.000146 |     | 0.000140 | 0.000146 | 0.000143 |      4.3 |     |                             | 0.000146 | 

$$ For HE1xx cases ABS[ (Max-Min) / (Analytic Solution)] and for HE2xx cases ABS[ (Max-Min) / (Mean of Example Simulation Results)]


# Table B16.6-4. Fan Energy, both fans (kWh)
| Case                       | ESP-r/HOT3000 | EnergyPlus | DOE-2.1E |     |   Min |   Max |  Mean | Dev % $$ |     | Analytical/Quasi-Analytical |  TEST | 
|:-------------------------- | -------------:| ----------:| --------:| ---:| -----:| -----:| -----:| --------:| ---:| ---------------------------:| -----:| 
| HE150 Continuous Circ. Fan |         432.0 |      433.3 |    432.1 |     | 432.0 | 433.3 |       |      0.3 |     |                       432.0 | 432.1 | 
| HE160 Cycling Circ. Fan    |         170.2 |      172.2 |    172.4 |     | 170.2 | 172.4 |       |      1.3 |     |                       172.8 | 172.4 | 
| HE170 Draft Fan            |         473.4 |      473.1 |    473.1 |     | 473.1 | 473.4 |       |      0.1 |     |                       473.2 | 473.1 | 
|                            | 
| HE210 Realistic Weather    |         281.6 |      291.4 |    298.9 |     | 281.6 | 298.9 | 290.6 |      6.0 |     |                             | 298.9 | 
| HE220 Setback Thermostat   |         268.3 |      276.1 |    281.2 |     | 268.3 | 281.2 | 275.2 |      4.7 |     |                             | 281.2 | 
| HE230 Undersized Furnace   |         458.3 |      431.4 |    478.4 |     | 431.4 | 478.4 | 456.0 |     10.3 |     |                             | 478.4 | 

$$ For HE1xx cases ABS[ (Max-Min) / (Analytic Solution)] and for HE2xx cases ABS[ (Max-Min) / (Mean of Example Simulation Results)]


# Table B16.6-5. Mean Zone Temperature (C)
| Case                     | ESP-r/HOT3000 | EnergyPlus | DOE-2.1E |     |   Min |   Max |  Mean | Dev % $$ |     | Analytical/Quasi-Analytical |  TEST | 
|:------------------------ | -------------:| ----------:| --------:| ---:| -----:| -----:| -----:| --------:| ---:| ---------------------------:| -----:| 
| HE210 Realistic Weather  |         20.01 |      20.00 |    19.98 |     | 19.98 | 20.01 | 20.00 |      0.2 |     |                             | 19.98 | 
| HE220 Setback Thermostat |         18.75 |      18.53 |    18.53 |     | 18.53 | 18.75 | 18.60 |      1.2 |     |                             | 18.53 | 
| HE230 Undersized Furnace |         15.48 |      15.17 |    15.64 |     | 15.17 | 15.64 | 15.43 |      3.0 |     |                             | 15.64 | 

$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]


# Table B16.6-6. Maximum Zone Temperature (C)
| Case                     | ESP-r/HOT3000 | EnergyPlus | DOE-2.1E |     |   Min |   Max |  Mean | Dev % $$ |     | Analytical/Quasi-Analytical |  TEST | 
|:------------------------ | -------------:| ----------:| --------:| ---:| -----:| -----:| -----:| --------:| ---:| ---------------------------:| -----:| 
| HE210 Realistic Weather  |         21.45 |      20.00 |    20.06 |     | 20.00 | 21.45 | 20.50 |      7.1 |     |                             | 20.06 | 
| HE220 Setback Thermostat |         22.70 |      20.00 |    20.11 |     | 20.00 | 22.70 | 20.94 |     12.9 |     |                             | 20.11 | 
| HE230 Undersized Furnace |         20.14 |      20.00 |    20.06 |     | 20.00 | 20.14 | 20.07 |      0.7 |     |                             | 20.06 | 

$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]


# Table B16.6-7. Minimum Zone Temperature (C)
| Case                     | ESP-r/HOT3000 | EnergyPlus | DOE-2.1E |     |   Min |   Max |  Mean | Dev % $$ |     | Analytical/Quasi-Analytical |  TEST | 
|:------------------------ | -------------:| ----------:| --------:| ---:| -----:| -----:| -----:| --------:| ---:| ---------------------------:| -----:| 
| HE210 Realistic Weather  |         20.00 |      20.00 |    19.89 |     | 19.89 | 20.00 | 19.96 |      0.6 |     |                             | 19.89 | 
| HE220 Setback Thermostat |         15.00 |      15.00 |    14.94 |     | 14.94 | 15.00 | 14.98 |      0.4 |     |                             | 14.94 | 
| HE230 Undersized Furnace |          1.45 |       4.48 |     3.22 |     |  1.45 |  4.48 |  3.05 |     99.3 |     |                             |  3.22 | 

$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]


