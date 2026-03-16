CREATE TABLE sentiment_jobs (
    id                  BIGSERIAL PRIMARY KEY,
    message_text        TEXT         NOT NULL,
    source_record_id    BIGINT,
    source_record_type  VARCHAR(100),

    status              VARCHAR(20)  NOT NULL DEFAULT 'pending',
    claimed_at          TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    claimed_by          VARCHAR(100) DEFAULT NULL,
    attempts            SMALLINT     NOT NULL DEFAULT 0,
    last_error          TEXT         DEFAULT NULL,

    frustration_label      VARCHAR(5)   DEFAULT NULL,
    satisfaction_label     VARCHAR(5)   DEFAULT NULL,
    detected_language      VARCHAR(10)  DEFAULT NULL,
    frustration_excerpt    TEXT         DEFAULT NULL,
    frustration_reasoning  TEXT         DEFAULT NULL,
    satisfaction_excerpt   TEXT         DEFAULT NULL,
    satisfaction_reasoning TEXT         DEFAULT NULL,
    raw_frustration_response TEXT       DEFAULT NULL,
    raw_satisfaction_response TEXT      DEFAULT NULL,

    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sentiment_jobs_pending
    ON sentiment_jobs (created_at) WHERE status = 'pending';

CREATE INDEX idx_sentiment_jobs_source
    ON sentiment_jobs (source_record_type, source_record_id);
