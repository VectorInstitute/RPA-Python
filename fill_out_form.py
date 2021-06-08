# Sample script to fill out a form
import rpa as r

r.init(headless_mode=True)

# r.url('https://ca.yahoo.com')
# r.type('//*[@id="ybar-sbq"]', 'github')

r.url('https://www.consumer.equifax.ca/personal/dispute-credit-report-form/')
r.wait(5.0)
r.fill_input("First Name", {'value': "Shayan"})
r.fill_input("Middle Name", {'value': "N/A"})
r.fill_input("Last Name", {'value': "Kousha"})
r.fill_input("Social Insurance Number", {'value': "abcdefghi"})
r.fill_input("Date of Birth", {
    'month': '6',
    'day': '5',
    'year': '2000'
})

# r.close()