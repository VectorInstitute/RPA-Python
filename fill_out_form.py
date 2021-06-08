# Sample script to fill out a form
import rpa as r

r.init(headless_mode=True)

r.url('https://www.consumer.equifax.ca/personal/dispute-credit-report-form/')
r.wait(5.0)
r.fill_input("First Name", "Shayan")

r.close()