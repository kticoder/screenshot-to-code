import asyncio
import os
import re
from openai import AsyncOpenAI
from bs4 import BeautifulSoup


async def process_tasks(prompts, api_key):
    tasks = [generate_image(prompt, api_key) for prompt in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"An exception occurred: {result}")
            processed_results.append(None)
        else:
            processed_results.append(result)

    return processed_results


async def generate_image(prompt, api_key):
    client = AsyncOpenAI(api_key=api_key)
    image_params = {
        "model": "dall-e-3",
        "quality": "standard",
        "style": "natural",
        "n": 1,
        "size": "1024x1024",
        "prompt": prompt,
    }
    res = await client.images.generate(**image_params)
    return res.data[0].url


def extract_dimensions(url):
    if matches := re.findall(r"(\d+)x(\d+)", url):
        width, height = matches[0]  # Extract the first match
        width = int(width)
        height = int(height)
        return (width, height)
    else:
        return (100, 100)


def create_alt_url_mapping(code):
    soup = BeautifulSoup(code, "html.parser")
    images = soup.find_all("img")

    return {
        image["alt"]: image["src"]
        for image in images
        if not image["src"].startswith("https://placehold.co")
    }


async def generate_images(code, api_key, image_cache):
    # Find all images
    soup = BeautifulSoup(code, "html.parser")
    images = soup.find_all("img")

    alts = [
        img.get("alt", None)
        for img in images
        if (
            img["src"].startswith("https://placehold.co")
            and image_cache.get(img.get("alt")) is None
        )
    ]
    # Exclude images with no alt text
    alts = [alt for alt in alts if alt is not None]

    # Remove duplicates
    prompts = list(set(alts))

    # Return early if there are no images to replace
    if not prompts:
        return code

    # Generate images
    results = await process_tasks(prompts, api_key)

    # Create a dict mapping alt text to image URL
    mapped_image_urls = dict(zip(prompts, results))

    # Merge with image_cache
    mapped_image_urls = {**mapped_image_urls, **image_cache}

    # Replace old image URLs with the generated URLs
    for img in images:
        # Skip images that don't start with https://placehold.co (leave them alone)
        if not img["src"].startswith("https://placehold.co"):
            continue

        if new_url := mapped_image_urls[img.get("alt")]:
            # Set width and height attributes
            width, height = extract_dimensions(img["src"])
            img["width"] = width
            img["height"] = height
            # Replace img['src'] with the mapped image URL
            img["src"] = new_url
        else:
            print("Image generation failed for alt text:" + img.get("alt"))

    # Return the modified HTML
    # (need to prettify it because BeautifulSoup messes up the formatting)
    return soup.prettify()
