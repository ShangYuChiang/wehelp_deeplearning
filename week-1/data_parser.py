import urllib.request
from html.parser import HTMLParser
import json
import csv

class ProductParser:
    def __init__(self, cateid: str, attr: str = "", pageCount: int = 40, page: int = 1):
        self.cateid = cateid
        self.attr = attr
        self.pageCount = pageCount
        self.page = page
        self.base_url = "https://ecshweb.pchome.com.tw/search/v4.3/all/results"

    def parser_all_product_ds(self):
        all_product_details = []
        page = self.page
        while True:
            query_params = f"?cateid={self.cateid}&attr={self.attr}&pageCount={self.pageCount}&page={page}"
            url = self.base_url + query_params
            #print(self.base_url + query_params)

            html_content = self.fetch_html(url)

            try:
                data = json.loads(html_content)
            except json.JSONDecodeError:
                #print(f"Unable to parse JSON content on page {page}")
                break

            products = data.get("Prods", [])
            if not products:
                #print(f"No product data on page {page}, stopping fetch")
                break

            for product in products:
                 all_product_details.append({
                    'Id': product.get('Id', 'N/A'),
                    'Name': product.get('Name', 'N/A'),
                    'Price': product.get('Price', 'N/A'),
                    'RatingValue': product.get('ratingValue', 'N/A'),
                    'ReviewCount': product.get('reviewCount', 'N/A')
        })
            page += 1
            #time.sleep(5)

        return all_product_details

    def fetch_html(self, url: str):
        try:
            with urllib.request.urlopen(url) as response:
                if response.status != 200:
                    raise RuntimeError(f"Request failed with status code {response.status}")
                return response.read().decode("utf-8")
        except Exception as e:
            raise RuntimeError("Fail to scrape web content") from e

class ProductProcessor:
    @staticmethod
    def process_write_to_file(filename: str, lines: list[str]) -> None:
        # Writes the provided lines to a file, each line separated by a newline character.
        with open(filename, "w", encoding="utf-8") as file:
            file.write("\n".join(lines))

    @staticmethod
    def process_best_products(all_prod_ds, min_Rating=4.9, min_Review=0):
        # Logic to process the best products
        best_products = [
            prod["Id"]
            for prod in all_prod_ds
            if prod.get("RatingValue") and prod["RatingValue"] > min_Rating and prod.get("ReviewCount", 0) > min_Review
        ]
        return best_products

    @staticmethod
    def process_average_price_i5(all_prod_ds):
        # Logic to process the average price for Intel i5 products
        i5_products = [
            prod["Price"]
            for prod in all_prod_ds
            if "i5" in prod["Name"]  # Check if the product name contains 'i5'
        ]
        if not i5_products:
            return 0.0
        return sum(i5_products) / len(i5_products)

    @staticmethod
    def process_zscore_for_price(all_prod_ds):
        # Logic to process the Z-score for product prices
        prices = [prod["Price"] for prod in all_prod_ds if "Price" in prod]
        mean = sum(prices) / len(prices) if prices else 0
        stddev = (sum((price - mean) ** 2 for price in prices) / len(prices)) ** 0.5 if prices else 1
        z_scores = [
            f"{prod['Id']},{prod['Price']},{(prod['Price'] - mean) / stddev:.2f}" for prod in all_prod_ds if "Price" in prod
        ]
        return z_scores


class TaskList:
    def __init__(self, Parser: ProductParser):
        self.parser = Parser
        self.all_prod_ds = self.parser.parser_all_product_ds()
 
    def task_1_product_id(self):
        product_ids = [prod["Id"] for prod in self.all_prod_ds]
        ProductProcessor.process_write_to_file("products.txt", product_ids)

    def task_2_best_products(self):
        best_product_ids = ProductProcessor.process_best_products(self.all_prod_ds)
        ProductProcessor.process_write_to_file("best-products.txt", best_product_ids)

    def task_3_average_price_i5(self):
        average_price = ProductProcessor.process_average_price_i5(self.all_prod_ds)
        print(f"{average_price}")

    def task_4_z_scores(self):
        z_scores = ProductProcessor.process_zscore_for_price(self.all_prod_ds)
        ProductProcessor.process_write_to_file("standardization.csv", z_scores)

def main():
    parser = ProductParser(cateid="DSAA31")
    task = TaskList(parser)
    task.task_1_product_id()
    task.task_2_best_products()
    task.task_3_average_price_i5()
    task.task_4_z_scores()

if __name__ == "__main__":
    main()
