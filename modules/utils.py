# modules/utils.py
import numpy as np
import pandas as pd
import QuantLib as ql
import scipy.stats as stats
import scipy.integrate as integrate
import re
import math
from datetime import datetime, timedelta
import calendar # For monthrange

# parsing functions
def parse_instrument_name(name):
    """
    Extract forward start and tenor from instrument name.
    """
    match = re.match(r'(\d+)([YM])(\d+)([YM])', name)
    if match:
        forward_value, forward_unit, tenor_value, tenor_unit = match.groups()
        forward_months = int(forward_value) * (12 if forward_unit == 'Y' else 1)
        tenor_months = int(tenor_value) * (12 if tenor_unit == 'Y' else 1)
        return forward_months, tenor_months
    return None, None

def parse_swaption_name(name):
    
    #Extract expiry years, tenor years, and strike from swaption name.
    
    parts = name.split('_')
    if len(parts) == 4 and parts[0] == 'Swaption':
        try:
            expiry_years = int(parts[1].replace('Y', ''))
            tenor_years = int(parts[2].replace('Y', ''))
            strike = float(parts[3])
            return expiry_years, tenor_years, strike
        except ValueError:
            return None, None, None
    return None, None, None

class Utils:
    @staticmethod
    def normal_cdf(x):
        return stats.norm.cdf(x)

    @staticmethod
    def integrate_variance(func, t, T0):
        result, _ = integrate.quad(func, t, T0)
        return result

    @staticmethod
    def H_function(a, t, T):
        if a == 0:
            return T - t
        return (1 - np.exp(-a * (T - t))) / a
    
    @staticmethod
    def year_fraction(date1, date2, day_count_convention='ACT/365'):
        if isinstance(date1, pd.Timestamp):
            date1 = date1.to_pydatetime()
        if isinstance(date2, pd.Timestamp):
            date2 = date2.to_pydatetime()

        if not isinstance(date1, datetime) or not isinstance(date2, datetime):
            raise TypeError("Inputs date1 and date2 must be datetime objects or Pandas Timestamps.")

        if day_count_convention == 'ACT/365':
            return (date2 - date1).days / 365.0
        elif day_count_convention == 'ACT/360':
            return (date2 - date1).days / 360.0
        elif day_count_convention == '30/360':
            y1, m1, d1 = date1.year, date1.month, date1.day
            y2, m2, d2 = date2.year, date2.month, date2.day
            if d1 == 31: d1 = 30
            if d2 == 31 and d1 == 30 : d2 = 30 # Adjusted original logic slightly
            elif d2 == 31 and d1 < 30 : d2 = 30 # Ensure if d1 is not 30/31, d2=31 becomes 30
            # Bond Basis (NASD method 1), assuming d1 is not end of Feb
            if m1 == 2 and d1 == calendar.monthrange(y1, 2)[1]: # d1 is end of Feb
                d1 = 30
            if m2 == 2 and d2 == calendar.monthrange(y2, 2)[1] and d1 == 30 : # d2 is end of Feb, and d1 was adjusted to 30 (or was 30)
                 d2 = 30


            return ( (y2 - y1) * 360.0 + (m2 - m1) * 30.0 + (d2 - d1) ) / 360.0
        else:
            raise ValueError(f"Day count convention not supported: {day_count_convention}")

    @staticmethod
    def bond_cashflows(K, payment_dates, day_count_convention='ACT/365', notional=1.0):
        n = len(payment_dates)
        if n == 0:
            return []
        
        # cashflows[j] corresponds to payment_dates[j]
        # The first element of payment_dates is T_0 (e.g., swap effective date)
        # Cashflows occur at T_1, T_2, ..., T_N
        cashflows_out = [0.0] * n # Initialize with zeros for all dates

        for i in range(1, n): # Iterate from the first payment period
            tau = Utils.year_fraction(payment_dates[i-1], payment_dates[i], day_count_convention)
            cashflows_out[i] = K * tau * notional

        if n > 0: # Add notional to the final cashflow at the last payment_date
            cashflows_out[-1] += notional
        
        return cashflows_out


    @staticmethod
    def generate_payment_dates(start_date, end_date, frequency):
        if frequency <= 0:
            raise ValueError("Frequency must be positive.")
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.to_pydatetime()
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.to_pydatetime()

        if start_date >= end_date: # If start date is after or same as end date, only return start date if it's the intended single payment date
            if start_date == end_date:
                 return [start_date]
            raise ValueError("Start date must be before end date for generating multiple payment dates.")


        payment_dates = [start_date]
        current_date = start_date
        
        if frequency == 0: # Should not happen due to check above, but as safeguard
            if start_date < end_date: payment_dates.append(end_date)
            return sorted(list(set(payment_dates)))

        months_step = 12 // frequency

        while True:
            next_month_raw = current_date.month + months_step
            year_increment = (next_month_raw - 1) // 12
            next_year = current_date.year + year_increment
            next_month = (next_month_raw - 1) % 12 + 1
            
            last_day_of_next_month = calendar.monthrange(next_year, next_month)[1]
            next_day = min(start_date.day, last_day_of_next_month) # Try to stick to original day of month
            
            candidate_date = datetime(next_year, next_month, next_day)

            if candidate_date <= end_date:
                payment_dates.append(candidate_date)
                current_date = candidate_date
                if current_date == end_date: # Reached end date exactly
                    break
            else: # Overshot end_date
                break
        
        # Ensure end_date is the last date if not already included
        if payment_dates[-1] < end_date:
            payment_dates.append(end_date)
        
        # Remove duplicates and ensure sorted order
        unique_dates = []
        seen = set()
        for d in sorted(payment_dates): # Sort before ensuring uniqueness
            if d not in seen:
                unique_dates.append(d)
                seen.add(d)
        return unique_dates
        
    @staticmethod
    def parse_swaption_ticker(ticker: str):
        """Parses a ticker like 'USSN0C10' into expiry and tenor in years."""
        tenor_years = int(ticker[-2:])
        expiry_code = ticker[4:-2]
        if expiry_code.endswith('C'): # 3 Months
            expiry_val = 0.25
        else:
            expiry_val = int(expiry_code)
        return expiry_val, tenor_years

    @staticmethod
    def bachelier_pricer(F, K, T, sigma_normal, df, is_payer=True):
        """Prices a European swaption using the Bachelier (Normal) model."""
        if T <= 0 or sigma_normal <= 0:
            if is_payer:
                return df * max(F - K, 0)
            else:
                return df * max(K - F, 0)

        d = (F - K) / (sigma_normal * math.sqrt(T))
        payer_price = df * ((F - K) * ql.CumulativeNormalDistribution()(d) + sigma_normal * math.sqrt(T) * ql.NormalDistribution()(d))
        
        if is_payer:
            return payer_price
        else:
            receiver_price = payer_price - df * (F - K)
            return receiver_price
    @staticmethod
    def calculate_forward_swap_rate(discount_curve, expiry_date, tenor_years, payment_frequency=1):
        """
        Calculates the par (at-the-money) forward swap rate.
        F = (P(T_start) - P(T_end)) / Annuity
        """
        # Define the swap schedule from the expiry date
        schedule_dates = [expiry_date]
        current_date = expiry_date
        months_step = 12 // payment_frequency
        for _ in range(tenor_years * payment_frequency):
            current_date = ql.TARGET().advance(current_date, ql.Period(months_step, ql.Months))
            schedule_dates.append(current_date)

        # Calculate the annuity (PV01 of the fixed leg)
        annuity = 0.0
        for i in range(1, len(schedule_dates)):
            tau = discount_curve.curve.dayCounter().yearFraction(schedule_dates[i-1], schedule_dates[i])
            df = discount_curve.discount(schedule_dates[i])
            annuity += tau * df

        if annuity < 1e-9:
            return 0.0

        # Calculate the value of the floating leg
        df_start = discount_curve.discount(expiry_date)
        df_end = discount_curve.discount(schedule_dates[-1])
        
        forward_rate = (df_start - df_end) / annuity
        return forward_rate