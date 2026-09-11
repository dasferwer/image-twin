CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE users(id uuid PRIMARY KEY,email text UNIQUE NOT NULL,password_hash text NOT NULL,role text NOT NULL DEFAULT 'user' CHECK(role IN ('user','admin')),created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE collections(id uuid PRIMARY KEY,user_id uuid NOT NULL REFERENCES users(id),name text NOT NULL,created_at timestamptz NOT NULL DEFAULT now(),UNIQUE(user_id,name));
CREATE TABLE images(
 id uuid PRIMARY KEY,collection_id uuid NOT NULL REFERENCES collections(id),pixel_sha256 text NOT NULL,source_sha256 text NOT NULL,
 object_key text NOT NULL UNIQUE,width integer NOT NULL,height integer NOT NULL,
 status text NOT NULL DEFAULT 'uploading' CHECK(status IN ('uploading','pending','queued','processing','ready','failed','deleted')),
 generation uuid,lease_until timestamptz,attempts integer NOT NULL DEFAULT 0,
 phash bit(64),embedding vector(1280),points jsonb,descriptors bytea,contrast double precision,encoder_version text,
 error text,created_at timestamptz NOT NULL DEFAULT now(),updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX images_pixels ON images(collection_id,pixel_sha256) WHERE status!='deleted';
CREATE INDEX images_collection ON images(collection_id,status);
CREATE INDEX images_pending ON images(created_at) WHERE status IN ('pending','queued','processing');
CREATE TABLE uploads(collection_id uuid NOT NULL REFERENCES collections(id),key text NOT NULL,request_sha256 text NOT NULL,image_id uuid NOT NULL REFERENCES images(id),PRIMARY KEY(collection_id,key));
CREATE TABLE image_events(id bigserial PRIMARY KEY,image_id uuid NOT NULL REFERENCES images(id),type text NOT NULL,created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE worker_heartbeats(name text PRIMARY KEY,updated_at timestamptz NOT NULL DEFAULT now());
