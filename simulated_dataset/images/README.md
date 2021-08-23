# Create 


## Clean images

### var
 * device = int
 * size_figure = int
 * action = str 
          => clean_images
          => dirty_images
 * start = int 
 * stop = int
 

```
cmd:  python3 create.py device size_figure action start stop
example: python3 create.py 0 28 clean_images 0 100
```

## Dirty images

### var
 * device = int
 * size_figure = int
 * action = str 
          => clean_images
          => dirty_images
 * start = int 
 * stop = int
 * type_psf = str
  
```
cmd:  python3 create.py device size_figure action start stop type_psf
example: python3 create.py 0 28 dirty_images 0 100 PSF_1
```

