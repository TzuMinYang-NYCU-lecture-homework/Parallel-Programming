#!/usr/bin/env python3

import getpass
import subprocess
import sys
from pathlib import Path

import pexpect

SSHDIR = Path.home() / ".ssh"
KEY_PATH = SSHDIR / "id_rsa"


def available_hosts():
    for i in range(2, 11):
        if i == 9:
            continue
        yield (f"pp{i}", f"192.168.202.{i}")


def generate_key():
    subprocess.run(
        ["ssh-keygen", "-t", "rsa", "-f", str(KEY_PATH), "-N", ""],
    )


def add_hosts():
    username = getpass.getuser()
    sshcfg = SSHDIR / "config"

    print("Writing host configurations...")

    with sshcfg.open("a+") as file:
        for hostname, ipaddr in available_hosts():
            print(f"Host {hostname}", file=file)
            print(f"\tHostName {ipaddr}", file=file)
            print(f"\tUser {username}", file=file)
            print(file=file)


def copy_keys(password: str):
    public_key_path = KEY_PATH.with_suffix(".pub")

    for hostname, _ in available_hosts():
        print(f"Copying public key into {hostname}")

        connection = pexpect.spawn(
            f"ssh-copy-id -i {public_key_path} {hostname}",
            encoding="utf-8",
        )

        ret = connection.expect(
            [
                pexpect.TIMEOUT,
                "Are you sure you want to continue connecting",
                "[pP]assword",
            ]
        )
        if ret == 0:
            print(f"Error connecting host {hostname}", file=sys.stderr)
            continue
        elif ret == 1:
            connection.sendline("yes")

            ret = connection.expect([pexpect.TIMEOUT, "[pP]assword"])

            if ret == 0:
                print(f"Error connecting host {hostname}", file=sys.stderr)
                continue
            elif ret == 1:
                connection.sendline(password)
        elif ret == 2:
            connection.sendline(password)

        connection.expect(pexpect.EOF)

        connection.close()


def main() -> None:
    password = getpass.getpass("SSH connection password: ")
    SSHDIR.mkdir(parents=True, exist_ok=True)
    generate_key()
    add_hosts()
    copy_keys(password)


if __name__ == "__main__":
    main()
