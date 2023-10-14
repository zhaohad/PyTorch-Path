import inspect

def dump(d):
    frame = inspect.currentframe()
    # https://blog.51cto.com/u_16175476/6793995
    try:
        for var_name, var_value in frame.f_back.f_locals.items():
            if var_value is d:
                d_name = var_name
                break;
    finally:
        del frame

    print(f"{d_name} = {d}, {d_name}.shape = {d.shape}")
