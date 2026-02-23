CREATE TABLE sentiment_jobs (
    id                  BIGSERIAL PRIMARY KEY,
    message_text        TEXT         NOT NULL,
    source_record_id    BIGINT,
    source_record_type  VARCHAR(100),
    language_hint       VARCHAR(10)  DEFAULT NULL,

    status              VARCHAR(20)  NOT NULL DEFAULT 'pending',
    claimed_at          TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    claimed_by          VARCHAR(100) DEFAULT NULL,
    attempts            SMALLINT     NOT NULL DEFAULT 0,
    last_error          TEXT         DEFAULT NULL,

    sentiment_label     VARCHAR(20)  DEFAULT NULL,
    excerpt             TEXT         DEFAULT NULL,
    reasoning           TEXT         DEFAULT NULL,
    raw_llm_response    TEXT         DEFAULT NULL,
    keyword_hits        JSONB        DEFAULT NULL,

    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sentiment_jobs_pending
    ON sentiment_jobs (created_at) WHERE status = 'pending';

CREATE INDEX idx_sentiment_jobs_source
    ON sentiment_jobs (source_record_type, source_record_id);
