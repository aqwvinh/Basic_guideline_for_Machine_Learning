# Use API

### Import library requests
```
import requests
from requests.auth import HTTPBasicAuth
```

Use `requests` library to be able to do REST method for API

Example with Aircall API
```
token = 'https://api.aircall.io/v1/calls' # API route
AIRCALL_TOKEN = "string1" # served as id
AIRCALL_API_KEY = 'string2' # served as password
start_date = '1630447200' #2021-09-01 in UNIX timestamp
end_date = '1633039200' #2021-10-01
per_page = 50
```

Check which method to use in the documentation. Here, it's a GET method so `requests.get`. 
After the API route, use `?` to pass parameters (here `from`, `to` and `per_page` params) and separate them by `&`
Use HTTPBasicAuth to use basic authentication (see documentation again)
```
# Get response
res = requests.get(token + f'?from={start_date}&to={end_date}&per_page={per_page}', auth=HTTPBasicAuth(AIRCALL_API_KEY, AIRCALL_TOKEN))
print(res)
```

