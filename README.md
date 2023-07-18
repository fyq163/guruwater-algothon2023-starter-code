# Brief

Leaderboard update submissions are due **9:00AM AEST 10 August.** 

Preliminary round submissions are due **9:00AM AEST 21 August.** 

Teams which proceed to the final round, will need to be available at **6:00PM AEST 6 September** to attend.

[//]: # (We recommend that each team have at least one member with programming experience in Python, as well as financial knowledge. Impressive submissions will be expected to involve reasonably advanced data analysis and implementation of reasonably sophisticated trading strategies.)

---

## Objective

Develop a trading strategy algorithm to perform optimally given certain metrics.

## How to Get Started

1. Assess provided price data from our simulated trading universe.
2. Build a predictive model.
3. Back-test the predictive model across given price data.
4. Evaluate your algorithmic strategy.
5. Hope for the best (just kidding!).
6. Consider factors such as optimisation and risk analysis. Topics that may be explored include:
   - Optimising for trade frequency.
   - Projections for worst-case scenarios and methods to mitigate this.
   - Considering risk factors, and techniques to minimise risk.

[//]: # (For more information, make sure to take a look at our [learning resources and technical advice]&#40;https://unsw-fintech-society-events.github.io/algothon2022/4resources/&#41; page!)

---

## **<p align="center"><ins>Case Brief</ins></p>**

## Task

Implement a function _getMyPosition()_ which

- Takes as input a NumPy array of the shape _nInst_ x _nt_.
  - nInst = 100 is the number of instruments.
  - nt is the number of days for which the prices have been provided.
- Returns a vector of desired positions.
  - i.e. This function returns a NumPy vector of integers. This integer denotes your daily position per instrument in the universe. With 100 instruments, we can expect this function to return 100 integers each time it is called.

## Data

All required data has been generated by us, and is available in the Github repo in the top right corner of this page. We'd highly recommend cloning this repo to use a base for your algorithm development, and to make it easier for submission.

- Our simulated trading universe consists of several years of daily price data, spanning 50 instruments.
- The instruments are numbered from 0 to 49, and days go chronologically from 0 onwards such that p[inst, t] indicates the price of the instrument inst on day t.
- The price data file contains a NumPy array of the shape nInst x nt.
- nInst = number of instruments, nt = number of days.

In the preliminary round, teams will be provided the first 250 days of price data to be used as training data. This can be found in prices.txt.

- There will be an interim leaderboard update. This round is optional and failure will not result in elimination from the competition. This update will assess your algorithm on days 250 - 499.
- Teams will then receive price data and results from the leaderboard evaluation.
- Preliminary round algorithms will be assessed on data from days 500 - 749.
- Successful teams will then receive price data and results from preliminary evaluation.
- Final round algorithms will be assessed on data from days 750 - 999.

## The Algorithm

### Format

Algorithms must be contained in a file titled _[teamName].py._

- This file must contain a function _getMyPosition()._
- _getMyPosition()_ must take in the daily price data, and output a vector of integer positions - the numbers of shares desired for each stock as the total final position after the last day.
- _getMyPosition()_ must be in the global scope of the file called _[teamName].py_ and have the appropriate signature.
  - The function will be called once a day, with the entire price history up to and including that day. For example, on day 240, your function should take as input an array of 100 inst x 240 days.
  - When _getMyPosition()_ is called, we will trade position differences from the previous position **at the most recent price, buying or selling.**
  - Consider the case where your last position was +30, and the new stock price is $20. If your new position is +100, _eval_ will register this as buying 70 **extra** shares at $20 a share. If your new position is -200, _eval_ will sell 230 shares also at $20 a share.

### **Accepted Packages**

To ensure that code runs smoothly on the servers used for marking, we advise the following:

- Use only standard packages and their respective versions from the Anaconda library. The best way to do this is to simply download Anaconda...
- Where necessary, packages that are not included as part of Anaconda, or version numbers that are greater than those provided in Anaconda need to be declared in the submission form in the relevant section.

We will attempt to import and run through all non-standard packages if declared. However, in the case that your code still does not run, your team will be **disqualified**. Similarly, if your submission does not declare a non-standard package or provide a brief description of its use, it will also be **disqualified**.

### **Considerations**

- A commission rate of 10 bps (0.0010) can be assumed, meaning you will be charged commission equating 0.0010 * totalDollarVolumeTraded. This will be deducted from your PL.
- Positions can be long or short (i.e. the integer positions can be either positive or negative).
- Teams are limited to a $10k position limit per stock, positive or negative. The $10k limit cannot be breached at the time of the trade.
  - This position limit may technically be exceeded in the case that exactly $10k worth of a stock is bought, and stock goes up the next day - this is fine.
  - However, given this occurs, the position must be slightly reduced to be no greater than $10k by the new day's price.
  - Note: _eval.py_ contains a function to clip positions to a maximum of $10k. This means that if the price is $10 and the algorithm returns a position of 1500 shares, _eval.py_ will assume a request of 1000 shares.

### **Assessment Benchmarks**

The program we will use to evaluate your algorithm is provided in _eval.py_

Metrics used to quantitatively assess your submission will include:

- PL (daily and mean),
- Return (net PL / _dollarVolumeTraded_),
- Sharpe Ratio, and
- Trading volume.

Your algorithms will be assessed against _unseen, future_ price data of the same 100 instruments within the provided simulated trading universe.

We expect algorithms to have a maximum runtime of ~10min.

[//]: # (## Submission)

[//]: # ()
[//]: # (Submission details can be found on our [Submission]&#40;https://unsw-fintech-society-events.github.io/algothon2022/5submission/&#41; page.)

[//]: # ()
[//]: # (Ensure that all code submitted is tested against _eval.py_ **This will be the test used to evaluate the performance of your algorithm.**)

[//]: # (Judging criteria can be found [here.]&#40;https://unsw-fintech-society-events.github.io/algothon2022/6criteria/&#41;)
