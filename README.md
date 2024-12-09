# Spiderer
## Description
Spiderer is a tool based on the A* algorithm that crawls a website and returns all URLs found within all acccessible pages. 
It is written in Python and uses the Requests library to make HTTP requests.
Tested on Python 3.10.12

## Installation
To run Spiderer, you will need to install its dependencies:
```
pip install -r requirements.txt
```

## Usage
### Basic usage
To use Spiderer, you will need to provide a starting URL or domain. 
```
python spiderer.py https://example.com 
```
### JSON output
You can output json using the -j option. If you provide a path, the output will be stored in a file as well.
The JSON output provides a lot more information than the default output.
```
python spiderer.py https://example.com -j output.json
```

### Remove external links
You can remove links that are not from the initial domain using the --remove-external or -re option.

```
python spiderer.py https://example.com --remove-external    
```

### Max iteration
You can set a maximum number of iterations using the --max-iteration or -i option.

```
python spiderer.py https://example.com --max-iteration 10
```

### Filters
You can filter found urls by extensions or/and status codes using the --allowed-extensions, --allowed-status-code, --excluded-extensions and --excluded-status-code options.

```
python spiderer.py https://example.com --allowed-extensions css js --allowed-status-code 200 --excluded-extensions txt --excluded-status-code 404 
```

### Cookies
You can provide cookies to requests using the --cookies option.

```
python spiderer.py https://example.com --cookies '{"cookie1": "value1", "cookie2": "value2"}'
```

### Headers
You can provide headers to requests using the --headers option.

```
python spiderer.py https://example.com --headers '{"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"}'
``` 

### Data
You can provide data to requests using the --data option.

```
python spiderer.py https://example.com --data '{"key1": "value1", "key2": "value2"}'
```

### Timeout
You can set a timeout for a single GET request using the --timeout option.

```
python spiderer.py https://example.com --timeout 10
``` 


### Verbose
You can enable verbose mode using the --verbose or -v option.

```
python spiderer.py https://example.com --verbose
```

## Roadmap
* Add multithreading
* Parse sitemaps and robots.txt

## Contributing
Contributions are welcome!
