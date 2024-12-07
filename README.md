# Spiderer
## Description
Spiderer is a tool based on the A* algorithm that crawls a website and returns all URLs found within all acccessible pages. 
It is written in Python and uses the Requests library to make HTTP requests.

## Installation
To install Spiderer, you will need to install its dependencies:
```
pip install -r requirements.txt
```

## Usage
To use Spiderer, you will need to provide a starting URL or domain. You can also specify a maximum depth for the crawler to go through.
```
python main.py https://example.com -d 2
```

You can output json using the -j option. If you provide a path, the output will be stored in a file as well.
```
python main.py https://example.com -j output.json
```

## Roadmap
* Add filter by extension 
* Add filter by status code
* Add multithreading
* Handles froms 
* Provides informations about the context where the link was found (.i.e forms)


## Contributing
Contributions are welcome!
