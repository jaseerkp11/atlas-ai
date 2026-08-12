"""TASK 001 compatibility wrapper — prefer `atlas.execution.mt5_client.MT5Client`."""

from atlas.execution.mt5_client import MT5Client


def connect_mt5():
    print("=" * 50)
    print("ATLAS AI - MT5 Connection Test")
    print("=" * 50)

    client = MT5Client()
    if not client.connect():
        print("Failed to initialize MT5 / paper client")
        return False

    account = client.account_info_dict()
    print("\nConnected Successfully!\n")
    print(f"Login      : {account.get('login')}")
    print(f"Server     : {account.get('server')}")
    print(f"Name       : {account.get('name')}")
    print(f"Balance    : {account.get('balance')}")
    print(f"Equity     : {account.get('equity')}")
    print(f"Leverage   : {account.get('leverage')}")

    from atlas.config import load_settings

    print(f"\nConfigured Symbols : {len(load_settings().symbols)}")
    client.disconnect()
    return True


if __name__ == "__main__":
    connect_mt5()
