def log_call(func):
    def wrapper(*args, **kwargs):
        print(f"\n🛠️  [LOG] Calling function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper
