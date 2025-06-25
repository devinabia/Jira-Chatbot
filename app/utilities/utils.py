import requests


class Utils:
    def shorten_url(url):
        try:
            api_url = f"http://tinyurl.com/api-create.php?url={url}"

            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                return response.text
            else:
                return url

        except:
            return url
