| Variable | Type | Description |
|---|---|---|
|u| `float` or `array[floats]`  | ocean surface relative wind speed (m/s) at height zu |
|t|`float` or `array[floats]`|  bulk air temperature (degC) at height zt|
| rh | `float` or `array[floats]` | relative humidity (%) at height zq |
| zu | `float` or `array[floats]`| wind sensor height (m)| 
| zt | `float` or `array[floats]` | bulk temperature sensor height (m) |
| zq | `float` or `array[floats]` |RH sensor height (m) |
| ts | `float` or `array[floats]` | sea water temperature (degC) (also see jcool) |
| P |  `float` or `array[floats]` | surface air pressure (mb)
| lat | `float` or `array[floats]` |  latitude (deg) |
| zi | `float` or `array[floats]` | PBL height (m) |
| Rs | `float` or `array[floats]` | downward shortwave radiation (W/m^2) |
| Rl | `float` or `array[floats]` | downward longwave radiation (W/m^2) |
| rain | `float` or `array[floats]` or `None` | rain rate (mm/hr) |
| cp | `float` or `array[floats]` or `None` | phase speed of dominant waves (m/s) |
| sigH | `float` or `array[floats]` or `None` | significant wave height (m) |
| jcool | `int` | cool skin option: <br > 1.  if Ts is bulk ocean temperature (default) <br > 2.   0 if Ts is ocean skin temperature |
| out | `str` |  Output variables to include (see [outputs](./c35_outputs.md) for more info) <br > 1. `'full'` includes all outputs <br > 2.  tau only outputs wind stress <br > 3. U10 only outputs 10 m adjusted wind speed
