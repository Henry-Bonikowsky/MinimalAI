#!/usr/bin/env python3
"""Deploy MinimalAI to BisectHosting via SFTP"""
import os
import sys
import glob
import argparse
import paramiko

HOST = "dedicatedny.gravelhost.com"
PORT = 2022
USER = "henrybonikowsky6cm.b4f712e6"
REMOTE_PATH = "./plugins/"

def get_sftp():
    password = os.environ.get("BISECT_PASS")
    if not password:
        print("Error: Set BISECT_PASS environment variable")
        sys.exit(1)
    transport = paramiko.Transport((HOST, PORT))
    transport.connect(username=USER, password=password)
    return paramiko.SFTPClient.from_transport(transport), transport

def cmd_deploy(args):
    """Upload latest JAR, optionally delete old ones"""
    jars = [j for j in glob.glob("build/libs/MinimalAI-*.jar") if "reobf" in j]
    if not jars:
        print("No reobf JAR found in build/libs/. Build first.")
        sys.exit(1)

    jar = max(jars, key=os.path.getmtime)
    jar_name = os.path.basename(jar)
    remote_path = REMOTE_PATH + jar_name

    print(f"Deploying: {jar}")
    print(f"To: sftp://{HOST}:{PORT}{remote_path}")

    sftp, transport = get_sftp()
    try:
        # Delete old JARs first
        if args.clean:
            for f in sftp.listdir(REMOTE_PATH):
                if f.startswith("MinimalAI-") and f.endswith(".jar") and f != jar_name:
                    print(f"Deleting old: {f}")
                    sftp.remove(REMOTE_PATH + f)

        sftp.put(jar, remote_path)
        print("Upload successful! Restart server to load new plugin.")
    finally:
        sftp.close()
        transport.close()

def cmd_ls(args):
    """List files on server"""
    path = args.path or REMOTE_PATH
    if not path.endswith('/'):
        path += '/'
    sftp, transport = get_sftp()
    try:
        for attr in sftp.listdir_attr(path):
            ftype = "d" if attr.st_mode & 0o40000 else "-"
            print(f"{ftype} {attr.st_size:>10}  {attr.filename}")
    finally:
        sftp.close()
        transport.close()

def cmd_cat(args):
    """Read file from server"""
    sftp, transport = get_sftp()
    try:
        with sftp.open(args.path, 'r') as f:
            print(f.read().decode('utf-8'))
    finally:
        sftp.close()
        transport.close()

def cmd_pull(args):
    """Download file from server to local"""
    sftp, transport = get_sftp()
    try:
        local_path = args.local or os.path.basename(args.remote)
        sftp.get(args.remote, local_path)
        print(f"Downloaded: {args.remote} -> {local_path}")
    finally:
        sftp.close()
        transport.close()

def cmd_push(args):
    """Upload file to server"""
    sftp, transport = get_sftp()
    try:
        remote_path = args.remote or REMOTE_PATH + os.path.basename(args.local)
        sftp.put(args.local, remote_path)
        print(f"Uploaded: {args.local} -> {remote_path}")
    finally:
        sftp.close()
        transport.close()

def cmd_rm(args):
    """Delete file on server"""
    sftp, transport = get_sftp()
    try:
        sftp.remove(args.path)
        print(f"Deleted: {args.path}")
    finally:
        sftp.close()
        transport.close()

def main():
    parser = argparse.ArgumentParser(description="BisectHosting SFTP tool")
    sub = parser.add_subparsers(dest="cmd")

    p_deploy = sub.add_parser("deploy", help="Upload latest JAR")
    p_deploy.add_argument("-c", "--clean", action="store_true", help="Delete old JARs first")

    p_ls = sub.add_parser("ls", help="List remote directory")
    p_ls.add_argument("path", nargs="?", help="Remote path (default: plugins dir)")

    p_cat = sub.add_parser("cat", help="Read remote file")
    p_cat.add_argument("path", help="Remote file path")

    p_pull = sub.add_parser("pull", help="Download file from server")
    p_pull.add_argument("remote", help="Remote file path")
    p_pull.add_argument("local", nargs="?", help="Local path (default: same name)")

    p_push = sub.add_parser("push", help="Upload file to server")
    p_push.add_argument("local", help="Local file path")
    p_push.add_argument("remote", nargs="?", help="Remote path (default: plugins dir)")

    p_rm = sub.add_parser("rm", help="Delete remote file")
    p_rm.add_argument("path", help="Remote file path")

    args = parser.parse_args()

    if args.cmd == "deploy" or args.cmd is None:
        if args.cmd is None:
            args.clean = False
        cmd_deploy(args)
    elif args.cmd == "ls":
        cmd_ls(args)
    elif args.cmd == "cat":
        cmd_cat(args)
    elif args.cmd == "pull":
        cmd_pull(args)
    elif args.cmd == "push":
        cmd_push(args)
    elif args.cmd == "rm":
        cmd_rm(args)

if __name__ == "__main__":
    main()
