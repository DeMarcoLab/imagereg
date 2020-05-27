# Guide to `imagereg` on MASSIVE

1. SSH into MASSIVE
```
ssh username@m3.massive.org.au
```

2. Request an interactive compute session with GPU
```
smux new-session --ntasks=2 --gres=gpu:1
```

If the interactive job doesn't start immediately, you can make a note of the job ID with `smux list-sessions`.

Once the job is available, you can connect to it with `smux attach-session <job_id>`

3. Load modules in the interactive job
Now you're in the interactive job with GPU available, we load the required modules and activate the `imagereg` conda environment.

```
module load cuda
module load anaconda
source activate /projects/eh55/conda_envs/imagereg

```

4. Start python or iPython, and run `imagereg`

Example use of the `imagereg` package:
```
import os
import imagereg

os.mkdir output_directory
imagereg.pipeline('tests/images', '*.tif', 'output_directory')
```
