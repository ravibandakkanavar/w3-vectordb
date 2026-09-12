"""Shared Supabase Postgres connection helper."""
import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    supabase_url = os.environ["SUPABASE_URL"]
    db_password  = os.environ["SUPABASE_DB_PASSWORD"]
    project_ref  = supabase_url.replace("https://", "").split(".")[0]
    return psycopg2.connect(
        host="aws-0-ap-northeast-2.pooler.supabase.com",
        port=5432,
        dbname="postgres",
        user=f"postgres.{project_ref}",
        password=db_password,
        sslmode="require",
        connect_timeout=30,
    )
