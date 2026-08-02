import MetaTrader5 as mt5


def connect_mt5():
    print("=" * 50)
    print("ATLAS AI - MT5 Connection Test")
    print("=" * 50)

    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        print("Error:", mt5.last_error())
        return False

    account = mt5.account_info()

    if account is None:
        print("❌ No MT5 account connected.")
        mt5.shutdown()
        return False

    print("\n✅ Connected Successfully!\n")

    print(f"Login      : {account.login}")
    print(f"Server     : {account.server}")
    print(f"Name       : {account.name}")
    print(f"Balance    : {account.balance}")
    print(f"Equity     : {account.equity}")
    print(f"Leverage   : {account.leverage}")
    print(f"Company    : {account.company}")

    symbols = mt5.symbols_get()

    print(f"\nAvailable Symbols : {len(symbols)}")

    mt5.shutdown()

    return True


if __name__ == "__main__":
    connect_mt5()