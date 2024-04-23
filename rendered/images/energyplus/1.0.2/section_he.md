# Section Thermal Fabric

# Table B16.6-1. Total Furnace Load (GJ)
| Case                       | ESP-r/HOT3000 | EnergyPlus |     |   Min |   Max |  Mean | Dev % $$ |     | DOE-2.1E | Analytical/Quasi-Analytical | 
|:-------------------------- | -------------:| ----------:| ---:| -----:| -----:| -----:| --------:| ---:| --------:| ---------------------------:| 
| HE100 100% eff.            |         77.94 |      77.75 |     | 77.75 | 77.94 |       |      0.2 |     |    77.75 |                       77.74 | 
| HE110 80% eff.             |         77.94 |      77.75 |     | 77.75 | 77.94 |       |      0.2 |     |    77.75 |                       77.74 | 
| HE120 80% eff., PLR=0.4    |         31.25 |      31.10 |     | 31.10 | 31.25 |       |      0.5 |     |    31.13 |                       31.10 | 
| HE130 No Load              |          0.00 |       0.00 |     |  0.00 |  0.00 |       |      0.0 |     |     0.15 |                        0.00 | 
| HE140 Periodic PLR         |         31.26 |      31.10 |     | 31.10 | 31.26 |       |      0.5 |     |    31.12 |                       31.10 | 
| HE150 Continuous Circ. Fan |         29.88 |      29.59 |     | 29.59 | 29.88 |       |      1.0 |     |    29.57 |                       29.65 | 
| HE160 Cycling Circ. Fan    |         31.26 |      30.46 |     | 30.46 | 31.26 |       |      2.6 |     |    30.49 |                       31.10 | 
| HE170 Draft Fan            |         29.88 |      29.59 |     | 29.59 | 29.88 |       |      1.0 |     |    29.57 |                       29.65 | 
|                            | 
| HE210 Realistic Weather    |         41.36 |      42.04 |     | 41.36 | 42.04 | 41.70 |      1.6 |     |    42.06 |                             | 
| HE220 Setback Thermostat   |         39.41 |      39.87 |     | 39.41 | 39.87 | 39.64 |      1.2 |     |    39.76 |                             | 
| HE230 Undersized Furnace   |         34.32 |      34.59 |     | 34.32 | 34.59 | 34.45 |      0.8 |     |    34.37 |                             | 

$$ For HE1xx cases ABS[ (Max-Min) / (Analytics Solution)] and for HE2xx cases ABS[ (Max-Min) / (Mean of Example Simulation Results)]


# Table B16.6-2. Total Furnace Input (GJ)
| Case                       | ESP-r/HOT3000 | EnergyPlus |     |   Min |   Max |  Mean | Dev % $$ |     | DOE-2.1E | Analytical/Quasi-Analytical | 
|:-------------------------- | -------------:| ----------:| ---:| -----:| -----:| -----:| --------:| ---:| --------:| ---------------------------:| 
| HE100 100% eff.            |         77.74 |      77.71 |     | 77.71 | 77.74 |       |      0.0 |     |    78.42 |                       77.71 | 
| HE110 80% eff.             |         96.92 |      97.22 |     | 96.92 | 97.22 |       |      0.3 |     |    98.02 |                       97.22 | 
| HE120 80% eff., PLR=0.4    |         38.41 |      38.27 |     | 38.27 | 38.41 |       |      0.4 |     |    38.56 |                       38.27 | 
| HE130 No Load              |          0.00 |       0.00 |     |  0.00 |  0.00 |       |      0.0 |     |     0.14 |                        0.00 | 
| HE140 Periodic PLR         |         39.00 |      39.00 |     | 39.00 | 39.00 |       |      0.0 |     |    38.76 |                       39.00 | 
| HE150 Continuous Circ. Fan |         37.23 |      36.94 |     | 36.94 | 37.23 |       |      0.8 |     |    36.82 |                       37.02 | 
| HE160 Cycling Circ. Fan    |         38.12 |      38.12 |     | 38.12 | 38.12 |       |      0.0 |     |    37.96 |                       38.09 | 
| HE170 Draft Fan            |         37.23 |      36.94 |     | 36.94 | 37.23 |       |      0.8 |     |    36.82 |                       37.02 | 
|                            | 
| HE210 Realistic Weather    |         50.53 |      52.01 |     | 50.53 | 52.01 | 51.27 |      2.9 |     |    52.37 |                             | 
| HE220 Setback Thermostat   |         47.87 |      49.35 |     | 47.87 | 49.35 | 48.61 |      3.0 |     |    49.47 |                             | 
| HE230 Undersized Furnace   |         41.37 |      42.55 |     | 41.37 | 42.55 | 41.96 |      2.8 |     |    43.22 |                             | 

$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]


# Table B16.6-3. Fuel Consumption (m3/2)
| Case                       | ESP-r/HOT3000 | EnergyPlus |     |      Min |      Max |     Mean | Dev % $$ |     | DOE-2.1E | Analytical/Quasi-Analytical | 
|:-------------------------- | -------------:| ----------:| ---:| --------:| --------:| --------:| --------:| ---:| --------:| ---------------------------:| 
| HE100 100% eff.            |      0.000263 |   0.000263 |     | 0.000263 | 0.000263 |          |      0.0 |     | 0.000265 |                    0.000263 | 
| HE110 80% eff.             |      0.000328 |   0.000329 |     | 0.000328 | 0.000329 |          |      0.3 |     | 0.000332 |                    0.000329 | 
| HE120 80% eff., PLR=0.4    |      0.000130 |   0.000130 |     | 0.000130 | 0.000130 |          |      0.4 |     | 0.000131 |                    0.000130 | 
| HE130 No Load              |      0.000000 |   0.000000 |     | 0.000000 | 0.000000 |          |        - |     | 0.000000 |                    0.000000 | 
| HE140 Periodic PLR         |      0.000132 |   0.000132 |     | 0.000132 | 0.000132 |          |      0.0 |     | 0.000131 |                    0.000132 | 
| HE150 Continuous Circ. Fan |      0.000126 |   0.000125 |     | 0.000125 | 0.000126 |          |      0.8 |     | 0.000125 |                    0.000125 | 
| HE160 Cycling Circ. Fan    |      0.000129 |   0.000129 |     | 0.000129 | 0.000129 |          |      0.0 |     | 0.000129 |                    0.000129 | 
| HE170 Draft Fan            |      0.000126 |   0.000125 |     | 0.000125 | 0.000126 |          |      0.8 |     | 0.000125 |                    0.000125 | 
|                            | 
| HE210 Realistic Weather    |      0.000171 |   0.000176 |     | 0.000171 | 0.000176 | 0.000173 |      2.9 |     | 0.000177 |                             | 
| HE220 Setback Thermostat   |      0.000162 |   0.000167 |     | 0.000162 | 0.000167 | 0.000164 |      3.0 |     | 0.000167 |                             | 
| HE230 Undersized Furnace   |      0.000140 |   0.000144 |     | 0.000140 | 0.000144 | 0.000142 |      2.8 |     | 0.000146 |                             | 

$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]


# Table B16.6-4. Fan Energy, both fans (kWh)
| Case                       | ESP-r/HOT3000 | EnergyPlus |     |   Min |   Max |  Mean | Dev % $$ |     | DOE-2.1E | Analytical/Quasi-Analytical | 
|:-------------------------- | -------------:| ----------:| ---:| -----:| -----:| -----:| --------:| ---:| --------:| ---------------------------:| 
| HE150 Continuous Circ. Fan |         432.0 |      433.3 |     | 432.0 | 433.3 |       |      0.3 |     |    432.1 |                       432.0 | 
| HE160 Cycling Circ. Fan    |         170.2 |      172.2 |     | 170.2 | 172.2 |       |      1.2 |     |    172.4 |                       172.8 | 
| HE170 Draft Fan            |         473.4 |      473.1 |     | 473.1 | 473.4 |       |      0.1 |     |    473.1 |                       473.2 | 
|                            | 
| HE210 Realistic Weather    |         281.6 |      291.4 |     | 281.6 | 291.4 | 286.5 |      3.4 |     |    298.9 |                             | 
| HE220 Setback Thermostat   |         268.3 |      276.1 |     | 268.3 | 276.1 | 272.2 |      2.9 |     |    281.2 |                             | 
| HE230 Undersized Furnace   |         458.3 |      431.4 |     | 431.4 | 458.3 | 444.9 |      6.0 |     |    478.4 |                             | 

$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]


# Table B16.6-5. Mean Zone Temperature (C)
| Case                     | ESP-r/HOT3000 | EnergyPlus |     |   Min |   Max |  Mean | Dev % $$ |     | DOE-2.1E | Analytical/Quasi-Analytical | 
|:------------------------ | -------------:| ----------:| ---:| -----:| -----:| -----:| --------:| ---:| --------:| ---------------------------:| 
| HE210 Realistic Weather  |         20.01 |      20.00 |     | 20.00 | 20.01 | 20.01 |      0.0 |     |    19.98 |                             | 
| HE220 Setback Thermostat |         18.75 |      18.53 |     | 18.53 | 18.75 | 18.64 |      1.2 |     |    18.53 |                             | 
| HE230 Undersized Furnace |         15.48 |      15.17 |     | 15.17 | 15.48 | 15.32 |      2.0 |     |    15.64 |                             | 

$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]


# Table B16.6-6. Maximum Zone Temperature (C)
| Case                     | ESP-r/HOT3000 | EnergyPlus |     |   Min |   Max |  Mean | Dev % $$ |     | DOE-2.1E | Analytical/Quasi-Analytical | 
|:------------------------ | -------------:| ----------:| ---:| -----:| -----:| -----:| --------:| ---:| --------:| ---------------------------:| 
| HE210 Realistic Weather  |         21.45 |      20.00 |     | 20.00 | 21.45 | 20.73 |      7.0 |     |    20.06 |                             | 
| HE220 Setback Thermostat |         22.70 |      20.00 |     | 20.00 | 22.70 | 21.35 |     12.6 |     |    20.11 |                             | 
| HE230 Undersized Furnace |         20.14 |      20.00 |     | 20.00 | 20.14 | 20.07 |      0.7 |     |    20.06 |                             | 

$$ ABS[ (Max-Min) / (Mean of Example Simulation Results)]


# Table B16.6-7. Minimum Zone Temperature (C)
| Case                     | ESP-r/HOT3000 | EnergyPlus |     |   Min |   Max |  Mean | Dev % $$ |     | DOE-2.1E | Analytical/Quasi-Analytical | 
|:------------------------ | -------------:| ----------:| ---:| -----:| -----:| -----:| --------:| ---:| --------:| ---------------------------:| 
| HE210 Realistic Weather  |         20.00 |      20.00 |     | 20.00 | 20.00 | 20.00 |      0.0 |     |    19.89 |                             | 
| HE220 Setback Thermostat |         15.00 |      15.00 |     | 15.00 | 15.00 | 15.00 |      0.0 |     |    14.94 |                             | 
| HE230 Undersized Furnace |          1.45 |       4.48 |     |  1.45 |  4.48 |  2.97 |    102.2 |     |     3.22 |                             | 

$$ For HE1xx cases ABS[ (Max-Min) / (Analytics Solution)] and for HE2xx cases ABS[ (Max-Min) / (Mean of Example Simulation Results)]


