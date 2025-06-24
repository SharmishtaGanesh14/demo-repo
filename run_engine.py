# ✅ Full Code: run_engine.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import time
import os
import glob
from core.aggregator import TimeframeAggregator
from core.fvg_tracker import FVGTracker
from core.swing_tracker import SwingTracker
from core.smc_shift_tracker_mss_filtered import StructureShiftTracker
from core.bos_logger import BOSLogger
from core.mss_logger import MSSLogger
from core.state_manager import StateManager
from core.ob_tracker import OrderBlockTracker
from core.sd_tracker import SDTracker
from core.fib_tracker import FibTracker
from core.ifvg_tracker import IFVGTracker
from core.breaker_fvg_tracker import BreakerFVGTracker
from indicators.ema_tracker import EMATracker
from indicators.vwap_tracker import VWAPTracker
from collections import defaultdict

def load_candles(filepath):
    df = pd.read_csv(filepath, parse_dates=['date'], dayfirst=True)
    if 'volume' not in df.columns:
        df['volume'] = 0
    return df.to_dict('records')

def clear_processed_data():
    folders = [
        "data/processed/fvgs",
        "data/processed/structure/mss",
        "data/processed/structure/bos",
        "data/processed/structure/fib",
        "data/processed/structure/std",
        "data/processed/bb",
        "data/processed/ifvgs",
        "data/processed/ob",
        "data/processed/swings",
        "data/processed/indicators",
    ]
    for folder in folders:
        for file in glob.glob(os.path.join(folder, "*.csv")):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Failed to remove {file}: {e}")

def run_engine(live_candles):
    clear_processed_data()
    aggregator = TimeframeAggregator()
    state = StateManager()
    timeframes = ['5m', '15m', '1h', '4h', '1d', '1w']

    modules = {
        'fvgs': {tf: FVGTracker(tf, state) for tf in timeframes},
        'swings': {tf: SwingTracker(tf, 6, state) for tf in timeframes},
        'structure': {
            tf: {
                'tracker': StructureShiftTracker(tf, state),
                'bos_logger': BOSLogger(tf),
                'mss_logger': MSSLogger(tf)
            } for tf in timeframes
        },
        'ema': {tf: EMATracker(tf, state) for tf in timeframes},
        'vwap': {tf: VWAPTracker(tf, state) for tf in timeframes},
        'obs': {tf: OrderBlockTracker(tf, state) for tf in timeframes},
        'sd': {tf: SDTracker(tf, state) for tf in timeframes},
        'fib': {tf: FibTracker(tf, state) for tf in timeframes},
        'ifvg': {tf: IFVGTracker(tf, state) for tf in timeframes},
        'breaker_fvg': {tf: BreakerFVGTracker(tf, state) for tf in timeframes},
    }

    candle_history = {tf: [] for tf in timeframes}
    perf_log = defaultdict(lambda: {'fvg': 0, 'swing': 0, 'structure': 0, 'ema': 0, 'vwap': 0, 'ob': 0, 'count': 0})

    start_time = time.perf_counter()

    for idx, candle in enumerate(live_candles):
        emitted = aggregator.add_candle('5m', candle)
        emitted['5m'] = candle

        for tf, tf_candle in emitted.items():
            candle_history[tf].append(tf_candle)

            t1 = time.perf_counter()
            modules['ema'][tf].add_candle(tf_candle)
            t2 = time.perf_counter()
            modules['vwap'][tf].add_candle(tf_candle)
            t3 = time.perf_counter()
            modules['fvgs'][tf].add_candle(tf_candle)
            t4 = time.perf_counter()
            modules['swings'][tf].add_candle(tf_candle)
            t5 = time.perf_counter()

            tracker = modules['structure'][tf]['tracker']
            bos_logger = modules['structure'][tf]['bos_logger']
            mss_logger = modules['structure'][tf]['mss_logger']

            swings = state.get_swings(tf)
            if not swings or len(swings) < 3:
                continue

            bos, mss = tracker.add_candle(tf_candle)
            if bos and bos.get("date"):
                bos_logger.append(bos)

            if mss and mss.get("date"):
                mss_logger.append(mss)
                modules['obs'][tf].add_mss_event(mss, candle_history[tf])
                modules['sd'][tf].add_mss_event(mss)
                modules['ifvg'][tf].add_mss_event(mss)
                modules['fib'][tf].update_fib(mss)
                modules['breaker_fvg'][tf].add_mss_event(mss)

            mss_post = tracker.update_mss_lifecycle(tf_candle)
            if mss_post and mss_post.get("date"):
                mss_logger.append(mss_post)
                modules['obs'][tf].add_mss_event(mss_post, candle_history[tf])
                modules['sd'][tf].add_mss_event(mss_post)
                modules['fib'][tf].update_fib(mss_post)

            modules['obs'][tf].check_pending_obs()
            modules['obs'][tf].check_mitigation(tf_candle)
            modules['breaker_fvg'][tf].evaluate_next_candle(tf_candle)

            t6 = time.perf_counter()

            perf_log[tf]['ema'] += (t2 - t1)
            perf_log[tf]['vwap'] += (t3 - t2)
            perf_log[tf]['fvg'] += (t4 - t3)
            perf_log[tf]['swing'] += (t5 - t4)
            perf_log[tf]['structure'] += (t6 - t5)
            perf_log[tf]['ob'] += (time.perf_counter() - t6)
            perf_log[tf]['count'] += 1

        if idx % 100 == 0:
            print(f"\n[{idx}] Performance breakdown (per candle, per timeframe):")
            for tf in timeframes:
                count = perf_log[tf]['count'] or 1
                print(
                    f"  {tf.upper()}: EMA {perf_log[tf]['ema']/count*1000:.2f} ms | "
                    f"VWAP {perf_log[tf]['vwap']/count*1000:.2f} ms | "
                    f"FVG {perf_log[tf]['fvg']/count*1000:.2f} ms | "
                    f"SWING {perf_log[tf]['swing']/count*1000:.2f} ms | "
                    f"STRUCTURE {perf_log[tf]['structure']/count*1000:.2f} ms | "
                    f"OB {perf_log[tf]['ob']/count*1000:.2f} ms"
                )

    total_time = time.perf_counter() - start_time
    print(f"\n[INFO] Total time taken to process {len(live_candles)} candles: {total_time:.2f} seconds\n")

    print("[INFO] Starting final flush of all modules...\n")
    for name, category in modules.items():
        for tf, mod in category.items():
            if isinstance(mod, dict):
                for subname, submod in mod.items():
                    if hasattr(submod, 'flush'):
                        print(f"Flushing: {name} → {tf} → {subname}")
                        submod.flush()
            else:
                if hasattr(mod, 'flush'):
                    print(f"Flushing: {name} → {tf}")
                    mod.flush()

    print(state.get_fib_levels('5m'))

if __name__ == "__main__":
    candles = load_candles("data/raw/market_data_5m_.csv")
    run_engine(candles)