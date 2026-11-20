-- railside 库（设计.md §4.2）。由 `railside db build` 从 数据/中间/<趟>/*.json、数据/成品/模板/v<n>/template.json、
-- 主数据/零件类型.csv、数据/中间/<趟>/agent_log.jsonl 删库重建；库不是真源，删了能重建。
--
-- 约定：
--   结果表都带 trip_id、template_ver、src、code_ver、created_at：
--     src         来龙去脉。'frames:<f0>-<f1>' 是原图帧范围（帧号从 1 起，含两端）；'file:<相对项目根的路径>' 是来源文件。
--     code_ver    产生这条结果的代码版本，取自来源 JSON 的 code_ver（git commit）。
--     created_at  来源 JSON 的生成时间（ISO 8601）。建库时间在 meta.built_at。
--   列名以 _json 结尾的存 JSON 文本，其余是数或文本。
--   坐标单位 px。趟坐标 (x, y) 按时间顺序、不翻转；模板坐标 (u, v)，u = 0 是车头（设计.md §2）。
--   外框拆成 x0,y0,x1,y1 / u0,v0,u1,v1（来源 JSON 里的 [x0, y0, x1, y1]），方便按位置查。
--   template_ver 形如 'v1'。

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,   -- built_at、code_ver（建库代码）、template_ver、trips_json、skipped_json、schema
    value TEXT
);

-- 导入过的每个文件：路径、内容哈希、它自带的 code_ver/created_at。用来核对库和文件是否同步。
CREATE TABLE source_file (
    path       TEXT PRIMARY KEY, -- 相对项目根
    trip_id    TEXT,             -- 模板和主数据为 NULL
    kind       TEXT,             -- trip、qc、manifest、calib、align、seg、parts、compare、template、part_type、agent_log
    sha256     TEXT,
    bytes      INTEGER,
    code_ver   TEXT,
    created_at TEXT
);

-- 模板版本（template.json）。trip_id 是建模板用的趟；head_x 是它的车头趟坐标，u = x - head_x。
CREATE TABLE template_version (
    template_ver TEXT PRIMARY KEY,
    trip_id      TEXT,
    head_x       INTEGER,
    n_anchors    INTEGER,
    n_parts      INTEGER,
    calib_json   TEXT,
    src          TEXT,
    code_ver     TEXT,
    created_at   TEXT
);

-- 零件类型主数据（主数据/零件类型.csv），阈值供对比和告警用。
CREATE TABLE part_type (
    type            TEXT PRIMARY KEY, -- 代号，如 BOLT
    name            TEXT,             -- 粗名
    parent_type     TEXT,
    desc            TEXT,
    ncc_min         REAL,
    shift_mm_max    REAL,
    dtheta_max_deg  REAL,
    area_change_max REAL,
    src             TEXT
);

-- 趟次：一行一趟，汇总 trip/qc/manifest/calib/align/seg 的要点。
-- template_ver 取 align.json 的 template_ver，没有时取建库时选的模板版本。
CREATE TABLE trip (
    trip_id             TEXT PRIMARY KEY,
    template_ver        TEXT,
    src_dir             TEXT,    -- 原图目录（相对项目根）
    camera              TEXT,
    n_frames            INTEGER,
    frame_w             INTEGER,
    frame_h             INTEGER,
    train_x0            INTEGER, -- 车内趟坐标范围 [train_x0, train_x1)
    train_x1            INTEGER,
    head_x              INTEGER,
    train_frame0        INTEGER,
    train_frame1        INTEGER,
    qc_verdict          TEXT,    -- pass / warn / fail
    qc_reasons_json     TEXT,
    qc_missing_json     TEXT,
    bad_seams           INTEGER,
    calib_source        TEXT,
    sy_mm_per_px        REAL,
    sx_mm_per_px_median REAL,
    calib_refs_json     TEXT,
    align_residual_json TEXT,    -- align.json 的 residual_px，原样
    align_u_trim0       REAL,
    n_align_anchors     INTEGER,
    seg_model_json      TEXT,
    n_seg_dets          INTEGER,
    src                 TEXT,
    code_ver            TEXT,    -- 取自 trip.json，没有时取 manifest.json
    created_at          TEXT
);

-- 面板（manifest.json 的 train_panels，每节车一块）。src 是面板覆盖的原图帧范围。
CREATE TABLE panel (
    trip_id      TEXT,
    panel        INTEGER,
    car          INTEGER,
    x0           INTEGER,
    x1           INTEGER,
    width_px     INTEGER,
    height_px    INTEGER,
    frame0       INTEGER,
    frame1       INTEGER,
    template_ver TEXT,
    src          TEXT,
    code_ver     TEXT,
    created_at   TEXT,
    PRIMARY KEY (trip_id, panel)
);

-- 定位靶。kind='template'：模板里的靶（trip_id 是模板趟，x = u + head_x，y = v）；
-- kind='align'：对齐时在本趟找到的靶（align.json 的 anchors，x/y 是本趟坐标，u/v 是模板坐标）。
CREATE TABLE anchor (
    id           INTEGER PRIMARY KEY,
    kind         TEXT,
    trip_id      TEXT,
    template_ver TEXT,
    anchor_id    TEXT,
    car          INTEGER,
    x            REAL,
    y            REAL,
    u            REAL,
    v            REAL,
    w            REAL,
    h            REAL,
    score        REAL,
    patch        TEXT,
    src          TEXT,
    code_ver     TEXT,
    created_at   TEXT,
    UNIQUE (kind, trip_id, template_ver, anchor_id)
);

-- 零件实例。kind='template'：模板里冻结的零件（template.json 的 parts，trip_id 是模板趟，没有 x 外框）；
-- kind='trip'：某趟找到的零件（parts.json 的 instances）。car/zone 模板件取自 template.json，趟内件从编码解析。
-- found_by：parent / pixel / topology / missing。src 是零件所在原图帧范围（parts.json 的 src_frames），没有时是来源文件。
CREATE TABLE part_instance (
    id           INTEGER PRIMARY KEY,
    kind         TEXT,
    trip_id      TEXT,
    template_ver TEXT,
    code         TEXT,
    type         TEXT,
    parent       TEXT,
    car          INTEGER,
    zone         TEXT,
    found_by     TEXT,
    match_score  REAL,
    u0 REAL, v0 REAL, u1 REAL, v1 REAL,
    x0 REAL, y0 REAL, x1 REAL, y1 REAL,
    crop         TEXT,  -- 局部图路径，原样
    mask         TEXT,  -- 掩膜路径，原样
    src          TEXT,
    code_ver     TEXT,
    created_at   TEXT,
    UNIQUE (kind, trip_id, template_ver, code)
);
CREATE INDEX part_instance_code ON part_instance (code);

-- 零件特征，长表：一个特征一行。数值放 value，非数值放 value_text（JSON）。
-- name 由来源字段展开：标量用字段名，数组加 [i]，对象加 .key，如 centroid_mm[0]、size_mm[1]、feat.hu[0]。
-- unit 按字段名推断：mm2、mm、deg、px，推不出为 NULL。
CREATE TABLE part_feature (
    id               INTEGER PRIMARY KEY,
    part_instance_id INTEGER REFERENCES part_instance (id),
    trip_id          TEXT,
    template_ver     TEXT,
    code             TEXT,
    name             TEXT,
    value            REAL,
    value_text       TEXT,
    unit             TEXT,
    src              TEXT,
    code_ver         TEXT,
    created_at       TEXT
);
CREATE INDEX part_feature_code ON part_feature (code, name);

-- 对比结果（compare.json 的 items，一个零件一行）。shift_mm 等指标是数时存数，是数组/对象时存 JSON 文本；
-- item_json 是原始条目全文。
CREATE TABLE comparison (
    id           INTEGER PRIMARY KEY,
    trip_id      TEXT,
    template_ver TEXT,
    code         TEXT,
    ncc          REAL,
    diff_area    REAL,
    shift_mm,
    dtheta_deg   REAL,
    n_alarms     INTEGER,
    item_json    TEXT,
    src          TEXT,
    code_ver     TEXT,
    created_at   TEXT
);
CREATE INDEX comparison_code ON comparison (code);

-- 告警：compare.json 每个 item 的 alarms[] 一条一行，加 foreign[] 每块一行（type='foreign'，code 为 NULL）。
-- alarms[] 的元素可以是字符串（就是 type）或对象（type 或 kind、value、threshold，其余进 detail_json）。
CREATE TABLE alarm (
    id            INTEGER PRIMARY KEY,
    comparison_id INTEGER REFERENCES comparison (id),
    trip_id       TEXT,
    template_ver  TEXT,
    code          TEXT,
    type          TEXT,
    value,
    threshold,
    u0 REAL, v0 REAL, u1 REAL, v1 REAL,
    detail_json   TEXT,
    src           TEXT,
    code_ver      TEXT,
    created_at    TEXT
);
CREATE INDEX alarm_trip_type ON alarm (trip_id, type);

-- 智能体任务日志。真源是 数据/中间/<趟>/agent_log.jsonl（每行 ts、trip、step、tool、args、result、status、code_ver）；
-- created_at = ts。
CREATE TABLE agent_log (
    id           INTEGER PRIMARY KEY,
    trip_id      TEXT,
    ts           TEXT,
    step         TEXT,
    tool         TEXT,
    args_json    TEXT,
    result_json  TEXT,
    status       TEXT,
    template_ver TEXT,
    src          TEXT,
    code_ver     TEXT,
    created_at   TEXT
);
CREATE INDEX agent_log_trip ON agent_log (trip_id, ts);
