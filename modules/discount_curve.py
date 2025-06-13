# modules/discount_curve.py
import pandas as pd
import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime 
from .utils import Utils, parse_instrument_name # Relative import

class DiscountCurve:
    """
    DiscountCurve class with robust curve construction methods.
    """
    def __init__(
        self,
        market_data: pd.DataFrame,
        fwd_start_months: list,
        tenor_months: list,
        calendar: ql.Calendar = ql.UnitedStates(ql.UnitedStates.NYSE),
        settlement_days: int = 2,
        fixed_frequency=ql.Annual,
        fixed_convention=ql.Unadjusted,
        fixed_daycount: ql.DayCounter = ql.Actual365Fixed(),
        float_index=ql.Eonia(),
        float_daycount: ql.DayCounter = ql.Actual365Fixed(),
        interp_daycount: ql.DayCounter = ql.Actual365Fixed()
    ):
        self.market_data = market_data
        self.fwd_start = fwd_start_months
        self.tenors = tenor_months
        self.calendar = calendar
        self.settlement_days = settlement_days
        self.fixed_frequency = fixed_frequency
        self.fixed_convention = fixed_convention
        self.fixed_daycount = fixed_daycount
        self.float_index = float_index
        self.float_daycount = float_daycount
        self.interp_daycount = interp_daycount
        self.curve = None
        self.valuation_date = None
        self.helpers = []


    def build_curve(self, valuation_date=None, verbose=True):
        """
        Creates a robust discount curve with instrument selection and bootstrapping.

        Args:
            valuation_date (pd.Timestamp or ql.Date, optional):
                Date for curve construction. Defaults to most recent date in market data.
            verbose : If True, print detailed information about curve construction.
        Returns:
            ql.PiecewiseLogLinearDiscount: Constructed discount curve
        """
        #storing the valuatino date as a pd.timestamp
        if isinstance(valuation_date, ql.Date):
            valuation_date = pd.Timestamp(valuation_date.year(), valuation_date.month(), valuation_date.dayOfMonth())
        self.valuation_date = valuation_date
        # Determine valuation date
        if valuation_date is None:
            valuation_date = self.market_data.index[-1]

        # Convert valuation date
        if isinstance(valuation_date, pd.Timestamp):
            vdate = ql.Date(valuation_date.day, valuation_date.month, valuation_date.year)
        else:
            vdate = valuation_date

        ql.Settings.instance().evaluationDate = vdate
        #vdate is ql.Date, valuation_date is pd.Timestamp

        # Get the latest available market conditions
        new_date = valuation_date;
        while not (valuation_date in self.market_data.index) :
            new_date = new_date - pd.Timedelta(days=1)
        rates = self.market_data.loc[new_date]

        # Extract instrument details
        instrument_details = []
        for name, rate in rates.items():
            fwd_months, tenor_months = parse_instrument_name(name)
            if fwd_months is not None and tenor_months is not None:
                instrument_details.append({
                    'name': name,
                    'rate': round(rate,6),
                    'forward_months': fwd_months,
                    'tenor_months': tenor_months
                })

        # Sort instruments by maturity/length
        instrument_details.sort(key=lambda x: (x['forward_months'], x['tenor_months']))

        # Create swap rate helpers
        helpers = []
        seen_pillars = set()

        for instrument in instrument_details:
            pillar = instrument['forward_months']+ instrument['tenor_months']

            # Skip duplicate pillars
            if pillar in seen_pillars:
                if verbose:
                    print(f"Skipping duplicate pillar: {instrument['name']}")
                continue

            seen_pillars.add(pillar)

            try:
                helper = ql.SwapRateHelper(
                    ql.QuoteHandle(ql.SimpleQuote(instrument['rate'])),
                    ql.Period(instrument['tenor_months'], ql.Months),
                    self.calendar,
                    self.fixed_frequency,
                    self.fixed_convention,
                    self.fixed_daycount,
                    self.float_index,
                    ql.QuoteHandle(),
                    ql.Period(instrument['forward_months'], ql.Months)
                )
                helpers.append(helper)
                if verbose:
                    print(f"Added instrument: {instrument['name']} (Rate: {instrument['rate']})")
            except Exception as e:
                if verbose:
                    print(f"Could not create helper for {instrument['name']}: {e}")
        self.helpers = helpers
        # Construct the curve
        try:
            curve = ql.PiecewiseLogLinearDiscount(vdate, helpers, self.interp_daycount)
            self.curve = curve  # Store the curve in the instance
            if verbose:
                print("Curve construction successful!")
            return curve
        except Exception as e:
            if verbose:
                print(f"Curve construction failed: {e}")
            return None
    
    def _to_ql_date(self, date) -> ql.Date:
        """Helper function to convert pandas Timestamps to QuantLib Dates."""
        if isinstance(date, pd.Timestamp):
            return ql.Date(date.day, date.month, date.year)
        return date

    def discount(self, date) -> float:
        """
        Calculates the discount factor for a given date.
        
        """
        ql_date = self._to_ql_date(date)
        return self.curve.discount(ql_date)

    def zero_rate(self, date, compounding=ql.Continuous, freq=ql.Annual) -> float:
        """
        Calculates the zero rate for a given date.
        
        """
        ql_date = self._to_ql_date(date)
        return self.curve.zeroRate(ql_date, self.interp_daycount, compounding, freq).rate()

    def forward_rate(self, date1, date2, compounding=ql.Continuous, freq=ql.Annual) -> float:
        """
        Calculates the forward rate between two dates.
        
        """
        ql_date1 = self._to_ql_date(date1)
        ql_date2 = self._to_ql_date(date2)
        return self.curve.forwardRate(ql_date1, ql_date2, self.interp_daycount, compounding, freq).rate()
        
    # modules/discount_curve.py

    def plot_curve(self, end_years: float = 30, points: int = 100):
        """
        Plots the zero yield curve and marks the pillar points.

        Args:
            end_years (float): The maximum number of years to display on the x-axis.
            points (int): The number of points to generate for the smooth curve line.
        """
        if self.curve ==None:
            print("Error: Curve has not been built yet. Please run build_curve() first.")
            return

        today = self.curve.referenceDate()
        years = [i * end_years / (points - 1) for i in range(points)]
        dates = [today + ql.Period(int(y * 365.25), ql.Days) for y in years]
        zeros = [self.zero_rate(d) * 100 for d in dates]
        pillar_years, pillar_zeros = [], []

        # Get the maturity date for each instrument helper
        pillar_dates = [helper.latestDate() for helper in self.curve.instruments()]
        pillar_zeros = [self.zero_rate(d) * 100 for d in pillar_dates]
        pillar_years = [self.interp_daycount.yearFraction(today, d) for d in pillar_dates]

        # 3. Plot both the curve and the pillar points using matplotlib
        plt.figure(figsize=(12, 7))
        plt.plot(years, zeros, label='Interpolated Zero Curve')
        if pillar_years:
            plt.plot(pillar_years, pillar_zeros, 'o', color='red', markersize=5, label='Pillar Points')
        plt.xlabel('Years to Maturity')
        plt.ylabel('Zero Rate (%)')
        plt.title(f'Bootstrapped Zero Yield Curve as of {today.ISO()}')
        plt.grid(True)
        plt.legend()
        plt.show()
        
    def validate_market_prices(self) -> pd.DataFrame:
            """
            Recalculates swap rates from the curve and compares them to market inputs.
            This confirms the bootstrap correctly repriced the input instruments.
            """
            if not self.curve or not self.helpers:
                print("Error: Curve has not been built and/or helper list is empty. Run build_curve() first.")
                return pd.DataFrame()
    
            results = []
            for i, helper in enumerate(self.helpers):
                market_rate = helper.quote().value()
                implied_rate = helper.impliedQuote()
                difference = market_rate - implied_rate                
                instrument_name = "Instrument " + str(i+1) 
    
                results.append({
                    "Instrument": instrument_name,
                    "Market Rate": market_rate,
                    "Implied Rate": implied_rate,
                    "Difference": difference
                })
    
            print("🕵️  Validation by Market Prices:")
            df = pd.DataFrame(results)
            if (df['Difference'].abs() < 1e-9).all():
                 print("✅ Success! The curve perfectly reprices all input instruments.")
            else:
                 print("❌ Warning! Significant differences found between market and implied rates.")
            return df

    def plot_all_curves(self, end_years: float = 30, points: int = 100):
            """
            Plots the zero-coupon curve and the instantaneous forward curve for visual analysis.
            """
            if not self.curve:
                print("Error: Curve has not been built yet. Please run build_curve() first.")
                return
    
            today = self.curve.referenceDate()
            years = np.linspace(0.01, end_years, points)
            dates = [today + ql.Period(int(y * 365.25), ql.Days) for y in years]          
            zeros = [self.zero_rate(d) * 100 for d in dates]
            forwards_inst = [self.forward_rate(d, d + ql.Period(1, ql.Days)) * 100 for d in dates]   #one day instentaneous forward
            pillar_years = [self.interp_daycount.yearFraction(today, h.latestDate()) for h in self.helpers]
            pillar_zeros = [self.zero_rate(h.latestDate()) * 100 for h in self.helpers]
            
            # Plotting
            plt.figure(figsize=(14, 7))
            plt.plot(years, zeros, label='Zero-Coupon Yield Curve', lw=2)
            plt.plot(years, forwards_inst, label='One-Day Forward Curve', linestyle='--', lw=2)
            plt.plot(pillar_years, pillar_zeros, 'o', color='red', markersize=6, label='Pillar Points')            
            plt.title(f'Curve Analysis as of {today.ISO()}', fontsize=16)
            plt.xlabel('Years', fontsize=12)
            plt.ylabel('Rate (%)', fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend()
            plt.show()


