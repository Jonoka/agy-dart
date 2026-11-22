#!/usr/bin/env node
/**
 * agy-dart PreInvocation hook:
 * Fast (<5ms) tail scan of transcript.jsonl for server-side KV-cache protection.
 * Uses Antigravity's zero-prefix-pollution UI-only `metadata.user_markdown`
 * (with empty `systemMessage: ""`) so alerts never mutate the model prompt prefix.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { enableVersionedCompileCache } from "./compile-cache.mjs";
import { getConfigSnapshot } from "../context/lib/config.js";

const CONTEXT_WARN_TOKENS = 130000;
const CONTEXT_CRITICAL_TOKENS = 160000;

function readStdin(timeoutMs = 1000) {
  return new Promise((resolve) => {
    let data = "";
    let settled = false;
    const onData = (chunk) => {
      data += chunk;
    };
    const done = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      process.stdin.off("data", onData);
      process.stdin.off("end", done);
      process.stdin.off("error", done);
      try {
        process.stdin.pause();
        process.stdin.unref?.();
      } catch {}
      resolve(data);
    };
    const timer = setTimeout(done, timeoutMs);
    timer.unref();
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", onData);
    process.stdin.on("end", done);
    process.stdin.on("error", done);
  });
}

function resolveSessionId(payload = {}) {
  return (
    payload.rootConversationId ||
    payload.root_conversation_id ||
    payload.parentConversationId ||
    payload.parent_conversation_id ||
    payload.conversationId ||
    payload.conversation_id ||
    payload.sessionId ||
    payload.session_id ||
    process.env.ANTIGRAVITY_ROOT_CONVERSATION_ID ||
    process.env.ANTIGRAVITY_PARENT_CONVERSATION_ID ||
    process.env.ANTIGRAVITY_CONVERSATION_ID ||
    "default"
  );
}

function resolveDirectConversationId(payload = {}) {
  return (
    payload.conversationId ||
    payload.conversation_id ||
    payload.sessionId ||
    payload.session_id ||
    process.env.ANTIGRAVITY_CONVERSATION_ID ||
    "default"
  );
}

function formatDuration(sec) {
  const sAll = Math.max(0, Math.round(Number(sec || 0)));
  if (sAll < 60) return `${sAll}s`;
  if (sAll >= 3600) {
    const h = Math.floor(sAll / 3600);
    const m = Math.floor((sAll % 3600) / 60);
    return m > 0 ? `${h}h ${m}m` : `${h}h`;
  }
  const m = Math.floor(sAll / 60);
  const s = sAll % 60;
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

function formatK(tokens) {
  return `${Math.round(tokens / 1000)}k`;
}

function formatUsdShort(usd) {
  const n = Number(usd);
  if (!Number.isFinite(n) || Math.abs(n) < 0.005) return "";
  return n < 0 ? `-$${Math.abs(n).toFixed(2)}` : `$${n.toFixed(2)}`;
}

function estimateColdReprefillHint(diag, isZh = false) {
  const pricePer1M = diag.inputPricePer1M ?? diag.modelProfile?.inputPer1M;
  const writeMult =
    Number.isFinite(diag.cacheWriteMultiplier) && diag.cacheWriteMultiplier >= 1
      ? diag.cacheWriteMultiplier
      : 1;
  const estColdUsd =
    Number.isFinite(pricePer1M) && pricePer1M > 0 && diag.currentContextTokens > 0
      ? (diag.currentContextTokens / 1e6) * pricePer1M * writeMult
      : 0;
  const coldUsdStr = formatUsdShort(estColdUsd);
  if (!coldUsdStr) return "";
  return isZh ? `（冷重算预计约 ${coldUsdStr}）` : ` (~${coldUsdStr} cold re-prefill)`;
}

export function evaluateCacheAlerts(diag, { sessionId, store, invocationNum = null, lang = null }) {
  const alerts = [];
  if (!diag || diag.turnsWithUsage === 0) return alerts;
  const resolvedLang =
    lang === "zh" || lang === "en"
      ? lang
      : typeof store?.getUiLang === "function"
        ? store.getUiLang()
        : "en";
  const isZh = resolvedLang === "zh";

  const parsedInvNum =
    invocationNum !== undefined && invocationNum !== null && invocationNum !== "" && typeof invocationNum !== "boolean"
      ? Number(invocationNum)
      : NaN;
  const isMidTurnContinuation =
    Number.isFinite(parsedInvNum) && parsedInvNum >= 1 && diag.inToolLoop !== false;

  const wmKey = `cache_alert:${sessionId}:watermark`;
  const ttlKey = `cache_alert:${sessionId}:ttl_turn`;
  const ttlExpiringKey = `cache_alert:${sessionId}:ttl_expiring_turn`;
  const mediaKey = `cache_alert:${sessionId}:media`;
  const loopKey = `cache_alert:${sessionId}:loop`;
  const getAlertMeta = (k, def = "") =>
    typeof store.getTimedMeta === "function" ? store.getTimedMeta(k, def) : store.getMeta(k, def);
  const setAlertMeta = (k, v) =>
    typeof store.setTimedMeta === "function" ? store.setTimedMeta(k, v) : store.setMeta(k, v);

  const prevWm = getAlertMeta(wmKey, "");
  const effectiveWarn = diag.warnTokens || CONTEXT_WARN_TOKENS;
  const effectiveCrit = diag.criticalTokens || CONTEXT_CRITICAL_TOKENS;
  const winSource = diag.windowSource || diag.modelProfile?.windowSource;
  const winSourceTag = winSource === "transcript" || winSource === "explicit" ? ` [${winSource}]` : "";
  const pctUsed = diag.modelProfile?.windowTokens
    ? Math.round((diag.contextUtilization || diag.currentContextTokens / diag.modelProfile.windowTokens) * 100)
    : 0;
  const windowLabel = diag.modelProfile?.windowTokens
    ? isZh
      ? `，占 ${formatK(diag.modelProfile.windowTokens)} 窗口 ${pctUsed}%${winSourceTag}`
      : `, ${pctUsed}% of ${formatK(diag.modelProfile.windowTokens)}${winSourceTag}`
    : "";
  const savedUsdStr = diag.estimatedSavedUsd >= 0.01 ? formatUsdShort(diag.estimatedSavedUsd) : "";
  const wastedUsdStr = diag.estimatedWastedUsd >= 0.01 ? formatUsdShort(diag.estimatedWastedUsd) : "";
  const needsHandoffCue =
    diag.compactAdvice === "critical_compact" || diag.compactAdvice === "opportunistic_cold_compact";

  // 1. Context Watermark alert (fires once per threshold crossing: ok -> warn -> critical)
  if (diag.watermark === "critical" && prevWm !== "critical") {
    setAlertMeta(wmKey, "critical");
    const hitPct = (diag.weightedHitRate * 100).toFixed(0);
    if (isZh) {
      const wasteHint = wastedUsdStr ? `，重填浪费约 ${wastedUsdStr}` : "";
      const handoffHint = needsHandoffCue ? "（建议先复制右侧面板「**交接简报**」）" : "";
      alerts.push(
        `当前上下文已达 **${formatK(diag.currentContextTokens)} tokens**（告急水位 >= ${formatK(effectiveCrit)}${windowLabel}，缓存命中率 ${hitPct}%${wasteHint}）。建议在当前任务阶段收尾后执行 \`/compact\`${handoffHint}，随后调用 \`ctx_resume\`。`
      );
    } else {
      const wasteHint = wastedUsdStr ? `, ~${wastedUsdStr} wasted` : "";
      const handoffHint = needsHandoffCue ? " (or copy Sidecar **Handoff Brief**)" : "";
      alerts.push(
        `Context at **${formatK(diag.currentContextTokens)} tokens** (critical watermark >= ${formatK(effectiveCrit)}${windowLabel}, cache hit ${hitPct}%${wasteHint}). Consider \`/compact\`${handoffHint} at the next milestone, then call \`ctx_resume\`.`
      );
    }
  } else if (diag.watermark === "warn" && prevWm !== "warn" && prevWm !== "critical") {
    if (diag.compactAdvice === "opportunistic_cold_compact" && isMidTurnContinuation) {
      // Suppress cold-compact handoff banner during mid-turn tool execution loops (invocationNum >= 1)
    } else {
      setAlertMeta(wmKey, "warn");
      const hitPct = (diag.weightedHitRate * 100).toFixed(0);
      if (diag.compactAdvice === "delay_hot_cache") {
        const tempRaw = String(diag.cacheTemperature || "warm").toLowerCase();
        if (isZh) {
          const tempZh = tempRaw === "hot" ? "热缓存" : "温缓存";
          const savedHint = savedUsdStr ? `，已省约 ${savedUsdStr}` : "";
          alerts.push(
            `当前上下文已达 **${formatK(diag.currentContextTokens)} tokens**（**${tempZh}**，命中率 ${hitPct}%${savedHint}${windowLabel}）。缓存正热时建议**暂缓 \`/compact\`** 以免触发全量冷重算；请继续通过 \`agy-dart_fast\` / \`ctx_execute\` 控制单次工具输出体积。`
          );
        } else {
          const tempLabel = tempRaw.toUpperCase();
          const savedHint = savedUsdStr ? `, ~${savedUsdStr} saved` : "";
          alerts.push(
            `Context reached **${formatK(diag.currentContextTokens)} tokens** (cache **${tempLabel}**, hit ${hitPct}%${savedHint}${windowLabel}). **Delay \`/compact\`** while cache is hot to avoid re-prefill; keep tool outputs compact via \`agy-dart_fast\` / \`ctx_execute\`.`
          );
        }
      } else if (diag.compactAdvice === "opportunistic_cold_compact") {
        alerts.push(
          isZh
            ? `当前上下文已达 **${formatK(diag.currentContextTokens)} tokens**（偏高水位 >= ${formatK(effectiveWarn)}${windowLabel}，缓存已冷）。当前适合复制右侧面板「**交接简报**」后执行 \`/compact\`，再调用 \`ctx_resume\`。`
            : `Context reached **${formatK(diag.currentContextTokens)} tokens** (high watermark >= ${formatK(effectiveWarn)}${windowLabel}, cache cold). Good moment for Sidecar **Handoff Brief** + \`/compact\`, then \`ctx_resume\`.`
        );
      } else {
        alerts.push(
          isZh
            ? `当前上下文已达 **${formatK(diag.currentContextTokens)} tokens**（偏高水位 >= ${formatK(effectiveWarn)}${windowLabel}，缓存命中率 ${hitPct}%）。请继续通过 \`agy-dart_fast\` / \`ctx_execute\` 保持输出精简。`
            : `Context reached **${formatK(diag.currentContextTokens)} tokens** (high watermark >= ${formatK(effectiveWarn)}${windowLabel}, cache hit ${hitPct}%). Keep outputs compact via \`agy-dart_fast\` / \`ctx_execute\`.`
        );
      }
    }
  } else if (diag.watermark === "ok" && prevWm) {
    // Reset watermark state after compaction
    store.deleteMeta(wmKey);
  }

  // 2. Idle Prompt-Cache TTL expiry & proactive expiring alerts (once per idle episode, turn entry only)
  const ttlMinContext = Math.max(60000, diag.coldCompactTokens || 60000);
  if (!isMidTurnContinuation) {
    if (diag.ttlStatus === "expired" && diag.currentContextTokens >= ttlMinContext && diag.lastPlannerAt) {
      const lastAlertedTurn = getAlertMeta(ttlKey, "");
      if (lastAlertedTurn !== diag.lastPlannerAt) {
        setAlertMeta(ttlKey, diag.lastPlannerAt);
        const costHint = estimateColdReprefillHint(diag, isZh);
        if (isZh) {
          const ttlSourceZh =
            diag.ttlSource === "empirical"
              ? "实证校准 "
              : diag.ttlSource === "explicit"
                ? "自定义 "
                : "";
          const compactTailZh = needsHandoffCue
            ? "若准备压缩，建议先复制右侧面板「**交接简报**」再执行 `/compact`，随后调用 `ctx_resume`。"
            : "若准备切换任务，适合顺手执行 `/compact`。";
          alerts.push(
            `已空闲 **${formatDuration(diag.idleSeconds)}**（超过 ${ttlSourceZh}${formatDuration(diag.cacheTtlSec)} 缓存 TTL），当前上下文 **${formatK(diag.currentContextTokens)} tokens**${costHint}——下一轮将触发冷重填。${compactTailZh}`
          );
        } else {
          const ttlSourceTag =
            diag.ttlSource === "empirical"
              ? " calibrated"
              : diag.ttlSource === "explicit"
                ? " configured"
                : "";
          const compactTail = needsHandoffCue
            ? "Good moment to copy Sidecar **Handoff Brief** + `/compact`, then `ctx_resume`."
            : "Good moment to `/compact` if switching tasks.";
          alerts.push(
            `Idle **${formatDuration(diag.idleSeconds)}** (>${formatDuration(diag.cacheTtlSec)}${ttlSourceTag} KV-cache TTL) at **${formatK(diag.currentContextTokens)} tokens**${costHint} — next turn will re-prefill cold. ${compactTail}`
          );
        }
      }
    } else if (diag.ttlStatus === "expiring" && diag.currentContextTokens >= ttlMinContext && diag.lastPlannerAt) {
      const lastAlertedExpiring = getAlertMeta(ttlExpiringKey, "");
      if (lastAlertedExpiring !== diag.lastPlannerAt) {
        setAlertMeta(ttlExpiringKey, diag.lastPlannerAt);
        const remSec = Math.max(0, Math.round(Number(diag.ttlRemainingSec ?? 0)));
        const remLabel = remSec <= 60 ? `${remSec}s` : formatDuration(remSec);
        const costHint = estimateColdReprefillHint(diag, isZh);
        alerts.push(
          isZh
            ? `**${formatK(diag.currentContextTokens)} tokens** 热缓存将在 **${remLabel}** 后过期${costHint}——尽快发送下一句可续热，或在过期后复制右侧面板「**交接简报**」执行 \`/compact\`。`
            : `KV-cache expiring in **${remLabel}** at **${formatK(diag.currentContextTokens)} tokens**${costHint} — send your next turn soon to stay warm, or copy Sidecar **Handoff Brief** before \`/compact\`.`
        );
      }
    }
  }

  // 3. Multimodal high-context TTFT cliff warning (once per session)
  if (diag.totalMediaCount > 0 && diag.currentContextTokens >= 150000) {
    const alreadyAlerted = getAlertMeta(mediaKey, "");
    if (!alreadyAlerted) {
      setAlertMeta(mediaKey, "1");
      alerts.push(
        isZh
          ? `当前 **${formatK(diag.currentContextTokens)} tokens** 会话中包含 **${diag.totalMediaCount} 个图片/多模态附件**（首字延迟可能放大 3–7 倍）。建议将截图检查委派给纯文本子代理或执行 \`/compact\`。`
          : `${diag.totalMediaCount} media/image item(s) active in a **${formatK(diag.currentContextTokens)}-token** session (3–7x TTFT cliff risk). Delegate screenshot checks to a text-only subagent or \`/compact\`.`
      );
    }
  }

  // 4. Doom-loop / edit<->fail oscillation convergence alert (fires once per escalation level, resets on convergence)
  const lw = diag.loopWatch;
  if (lw && (lw.status === "warn" || lw.status === "critical")) {
    const prevLoop = getAlertMeta(loopKey, "");
    const shouldEmit =
      (lw.status === "critical" && prevLoop !== "critical") ||
      (lw.status === "warn" && prevLoop !== "warn" && prevLoop !== "critical");
    if (shouldEmit) {
      setAlertMeta(loopKey, lw.status);
      const details = [];
      if (lw.editFailOscillations >= 2) {
        const hasFiles = Array.isArray(lw.oscillatingFiles) && lw.oscillatingFiles.length > 0;
        if (isZh) {
          const filesPartZh = hasFiles ? `（\`${lw.oscillatingFiles.join("`, `")}\`）` : "";
          details.push(`**${lw.editFailOscillations} 次改错↔失败震荡**${filesPartZh}`);
        } else {
          const filesPart = hasFiles ? ` on \`${lw.oscillatingFiles.join("`, `")}\`` : "";
          details.push(`**${lw.editFailOscillations}x edit↔fail oscillation**${filesPart}`);
        }
      }
      const hasRepeat = lw.reason === "repeat_loop" || lw.reason === "repeat_and_oscillation";
      if (hasRepeat && lw.maxRepeatCount >= 3) {
        if (isZh) {
          const callPartZh = lw.topRepeatedCall ? `（\`${lw.topRepeatedCall}\` ×${lw.maxRepeatCount}）` : `（×${lw.maxRepeatCount}）`;
          details.push(`**${Math.round((lw.repeatRate || 0) * 100)}% 重复工具调用**${callPartZh}`);
        } else {
          const callPart = lw.topRepeatedCall ? ` (\`${lw.topRepeatedCall}\` ×${lw.maxRepeatCount})` : ` (×${lw.maxRepeatCount})`;
          details.push(`**${Math.round((lw.repeatRate || 0) * 100)}% tool repeat rate**${callPart}`);
        }
      }
      alerts.push(
        isZh
          ? `死循环收敛监控（${lw.status.toUpperCase()}）：${details.join(" + ")}。建议暂停重复打补丁，重新核查根因或切换策略。`
          : `Convergence watch (${lw.status.toUpperCase()}): ${details.join(" + ")}. Pause repetitive patching, re-check root cause/assumptions, or pivot strategy.`
      );
    }
  } else if (lw && lw.status === "ok" && getAlertMeta(loopKey, "")) {
    store.deleteMeta(loopKey);
  }

  return alerts;
}

async function main() {
  const cfgSnapshot = getConfigSnapshot();
  const effectiveCfg = cfgSnapshot.values;
  const resolvedDbPath = effectiveCfg.dbPath || path.join(os.homedir(), ".agy-dart", "context.db");
  const watchdogExplicit = effectiveCfg.stallWatchdog && cfgSnapshot.sources.stallWatchdog !== "default";
  const ttlNotifyExplicit = effectiveCfg.ttlDesktopNotify && cfgSnapshot.sources.ttlDesktopNotify !== "default";
  const dbExists = fs.existsSync(resolvedDbPath);
  const needExistingDbWork =
    !effectiveCfg.cacheWatch &&
    !effectiveCfg.diffRead &&
    (effectiveCfg.readDedup || effectiveCfg.stallWatchdog || effectiveCfg.ttlDesktopNotify || dbExists) &&
    dbExists;
  if (!effectiveCfg.cacheWatch && !effectiveCfg.diffRead && !watchdogExplicit && !ttlNotifyExplicit && !needExistingDbWork) {
    process.stdout.write("{}\n");
    return;
  }

  const raw = (await readStdin()).trim();
  if (!raw) {
    process.stdout.write("{}\n");
    return;
  }

  let payload;
  try {
    payload = JSON.parse(raw);
  } catch {
    process.stdout.write("{}\n");
    return;
  }

  const sessionId = resolveSessionId(payload);
  const directConvId = resolveDirectConversationId(payload);
  enableVersionedCompileCache();

  let store = null;

  try {
    const [storeMod, cacheAnalyzerMod, watchdogLeaseMod] = await Promise.all([
      import("../context/lib/store.js"),
      import("../context/lib/cache-analyzer.js"),
      effectiveCfg.stallWatchdog || effectiveCfg.ttlDesktopNotify || dbExists
        ? import("../context/lib/watchdog-lease.js")
        : Promise.resolve(null)
    ]);
    const ContextModeStore = storeMod.ContextModeStore;

    // Hook mode: bounded SQLite busy timeout & deadline so external write locks fail-open in <1s
    store = new ContextModeStore({ hookMode: true, maxBusyRetries: 4, maxRetryDeadlineMs: 800 });

    const { analyzeTranscriptCache, findTranscriptForConversation, isSlashCompactText, readTranscriptTail } =
      cacheAnalyzerMod;

    const targetConvId = directConvId && directConvId !== "default" ? directConvId : sessionId;
    const isSubagentConv = Boolean(directConvId && directConvId !== "default" && directConvId !== sessionId);

    // Advance per-conversation turn sequence on every LLM turn when readDedup or diffRead is active
    // so non-read turns (edits/commands/chat) accurately age read-dedup & diff-read baselines — even if
    // cacheWatch is disabled or alerts are paused (PreInvocation is the sole turn clock; read_file never bumps it).
    let currentTurnSeq = 0;
    if ((effectiveCfg.diffRead || effectiveCfg.readDedup) && directConvId && directConvId !== "default") {
      try {
        currentTurnSeq = store.bumpConversationTurnSeq(directConvId);
      } catch {}
    }

    // Parse initialNumSteps & invocationNum for rollback detection and idempotent `/compact` invalidation
    const rawInitialSteps = payload.initialNumSteps ?? payload.initial_num_steps;
    const initialNumSteps =
      rawInitialSteps !== undefined &&
      rawInitialSteps !== null &&
      rawInitialSteps !== "" &&
      typeof rawInitialSteps !== "boolean" &&
      Number.isFinite(Number(rawInitialSteps)) &&
      Number(rawInitialSteps) >= 0
        ? Math.floor(Number(rawInitialSteps))
        : null;
    const rawInvNum = payload?.invocationNum ?? payload?.invocation_num ?? null;
    const parsedInvNum =
      rawInvNum !== undefined && rawInvNum !== null && rawInvNum !== "" && typeof rawInvNum !== "boolean"
        ? Number(rawInvNum)
        : NaN;
    const isMidTurnContinuation = Number.isFinite(parsedInvNum) && parsedInvNum >= 1;
    const isExplicitTurnEntry = Number.isFinite(parsedInvNum) && parsedInvNum === 0;

    // 1. Detect initialNumSteps rollback (trajectory rewind/reset -> immediate invalidation)
    if (initialNumSteps !== null && targetConvId && targetConvId !== "default") {
      try {
        const stepsKey = `compact_invalidated:steps:${targetConvId}`;
        const rawPrevSteps = store.getTimedMeta(stepsKey, "");
        const prevSteps = rawPrevSteps !== "" && Number.isFinite(Number(rawPrevSteps)) ? Math.floor(Number(rawPrevSteps)) : null;
        if (prevSteps !== null && initialNumSteps < prevSteps) {
          store.invalidateFileReads(targetConvId, null);
          store.deleteMeta(`compact_invalidated:tx_stat:${targetConvId}`);
          store.deleteMeta(`compact_invalidated:cmd:${targetConvId}`);
        }
        if (prevSteps !== initialNumSteps) {
          store.setTimedMeta(stepsKey, String(initialNumSteps));
        }
      } catch {}
    }

    // 2. Immediate `/compact` command detection via lastUserInput (idempotent via initialNumSteps;
    //    never re-invalidates during mid-turn tool loops where invocationNum >= 1).
    const rawLastUserInput = payload.lastUserInput ?? payload.last_user_input ?? "";
    if (typeof rawLastUserInput === "string" && rawLastUserInput.trim() && targetConvId && targetConvId !== "default") {
      try {
        const cmdKey = `compact_invalidated:cmd:${targetConvId}`;
        if (isSlashCompactText(rawLastUserInput)) {
          if (!isMidTurnContinuation) {
            const cmdText = rawLastUserInput.trim().slice(0, 128);
            let prevCmd = null;
            const rawPrevCmd = store.getTimedMeta(cmdKey, "");
            if (rawPrevCmd) {
              try {
                prevCmd = JSON.parse(rawPrevCmd);
              } catch {}
            }
            const alreadyStamped = Boolean(
              prevCmd &&
                prevCmd.text === cmdText &&
                (initialNumSteps !== null ? prevCmd.initialNumSteps === initialNumSteps : !isExplicitTurnEntry)
            );
            if (!alreadyStamped) {
              store.invalidateFileReads(targetConvId, null);
              store.setTimedMeta(
                cmdKey,
                JSON.stringify({
                  text: cmdText,
                  initialNumSteps,
                  stepIdx: null,
                  turnSeq: currentTurnSeq,
                  ts: Date.now()
                })
              );
            }
          }
        } else {
          // Non-compact user turn resets cmd stamp so a subsequent `/compact` always triggers cleanly
          if (store.getTimedMeta(cmdKey, "")) {
            store.deleteMeta(cmdKey);
          }
        }
      } catch {}
    }

    let transcriptPath =
      (typeof payload.transcriptPath === "string" && payload.transcriptPath.trim()) ||
      (typeof payload.transcript_path === "string" && payload.transcript_path.trim()) ||
      "";
    if (!transcriptPath || !fs.existsSync(transcriptPath)) {
      if (targetConvId && targetConvId !== "default") {
        transcriptPath = findTranscriptForConversation(targetConvId, { exactOnly: true }) || "";
      }
    }

    let diag = null;
    let loadedSteps = [];
    let txStatMtimeMs = 0;
    let txStatSize = 0;
    if (transcriptPath && fs.existsSync(transcriptPath)) {
      let txStatSig = "";
      try {
        const txStat = fs.statSync(transcriptPath);
        if (txStat.isFile() && txStat.size > 0) {
          txStatMtimeMs = txStat.mtimeMs;
          txStatSize = txStat.size;
          txStatSig = `${txStat.size}:${Math.round(txStat.mtimeMs)}:${transcriptPath}`;
        }
      } catch {}

      const steps = readTranscriptTail(transcriptPath, 131072, { minUsageSteps: 2 });
      loadedSteps = steps;
      if (steps.length > 0) {
        const diagOpts = {
          conversationId: targetConvId && targetConvId !== "default" ? targetConvId : ""
        };
        if (effectiveCfg.cacheTtlSec !== null) diagOpts.ttlSec = effectiveCfg.cacheTtlSec;
        if (effectiveCfg.windowTokens !== null) diagOpts.windowTokens = effectiveCfg.windowTokens;
        if (effectiveCfg.warnTokens !== null) diagOpts.warnTokens = effectiveCfg.warnTokens;
        if (effectiveCfg.criticalTokens !== null) diagOpts.criticalTokens = effectiveCfg.criticalTokens;
        if (dbExists || store.db) {
          try {
            diagOpts.historicalTtlByProfile = store.getAllModelTtlCalibrations();
          } catch {}
        }
        diag = analyzeTranscriptCache(steps, diagOpts);

        if (
          diag.ttlObservations &&
          Object.keys(diag.ttlObservations).length > 0 &&
          targetConvId &&
          targetConvId !== "default"
        ) {
          try {
            const prevTxSig = store.getTimedMeta(`compact_invalidated:tx_stat:${targetConvId}`, "");
            if (!txStatSig || prevTxSig !== txStatSig) {
              store.recordModelTtlObservations(diag.ttlObservations, targetConvId);
            }
          } catch {}
        }

        // Auto-invalidate file_reads & file_snapshots on any compaction/pruning event even when cacheWatch is off or paused.
        // Uses unified latestCompactionEvent (compaction bust, decoupled shrink / mild prune, explicit compact step, `/compact`)
        // and preserves any post-compaction reads when explicit stepIndex is known.
        const compactEvent = diag.latestCompactionEvent;
        if (compactEvent && compactEvent.stamp) {
          const lastCompactKey = `compact_invalidated:${targetConvId}`;
          if (store.getTimedMeta(lastCompactKey, "") !== compactEvent.stamp) {
            if (compactEvent.hasExplicitStepIndex && Number.isFinite(compactEvent.stepIndex)) {
              store.invalidateFileReadsBefore(targetConvId, { maxStep: compactEvent.stepIndex });
            } else {
              store.invalidateFileReads(targetConvId, null);
            }
            store.setTimedMeta(lastCompactKey, compactEvent.stamp);
          }
        }
        if (txStatSig && targetConvId && targetConvId !== "default") {
          try {
            store.setTimedMeta(`compact_invalidated:tx_stat:${targetConvId}`, txStatSig);
          } catch {}
        }
      }
    }

    // Record Sidecar Stall Watchdog / Idle TTL execution lease (isolated by direct conversationId)
    const watchdogActive = Boolean(
      effectiveCfg.stallWatchdog &&
        (typeof store.isWatchdogEnabled !== "function" || store.isWatchdogEnabled())
    );
    const ttlNotifyActive = Boolean(
      effectiveCfg.ttlDesktopNotify ||
        ((dbExists || store.db) &&
          typeof store.isTtlDesktopNotifyEnabled === "function" &&
          store.isTtlDesktopNotifyEnabled())
    );
    if (
      (watchdogActive || ttlNotifyActive) &&
      targetConvId &&
      targetConvId !== "default" &&
      watchdogLeaseMod?.recordHookLeaseTransition
    ) {
      try {
        let detectedSubagent = isSubagentConv;
        if (!detectedSubagent) {
          if (
            watchdogLeaseMod.isSubagentLeasePayload?.(payload) ||
            (typeof rawLastUserInput === "string" && watchdogLeaseMod.isHeadlessWorkerPrompt?.(rawLastUserInput))
          ) {
            detectedSubagent = true;
          } else if (loadedSteps.length > 0) {
            for (const s of loadedSteps) {
              if (!s || typeof s !== "object") continue;
              const stType = String(s.type || "").toUpperCase();
              if (stType === "USER_INPUT" || String(s.role || "").toLowerCase() === "user") {
                const uTxt =
                  typeof s.user_input === "string"
                    ? s.user_input
                    : typeof s.content === "string"
                      ? s.content
                      : "";
                if (uTxt && watchdogLeaseMod.isHeadlessWorkerPrompt?.(uTxt)) {
                  detectedSubagent = true;
                  break;
                }
              }
              if (Array.isArray(s.tool_calls)) {
                const hasFinish = s.tool_calls.some((c) => {
                  const n = c && typeof c === "object" ? c.name || c.tool || c.function?.name || "" : "";
                  return watchdogLeaseMod.extractBareToolName?.(n) === "finish";
                });
                if (hasFinish) {
                  detectedSubagent = true;
                  break;
                }
              }
            }
          }
        }
        watchdogLeaseMod.recordHookLeaseTransition(store, "PreInvocation", payload, {
          directConvId: targetConvId,
          rootSessionId: sessionId,
          isSubagent: detectedSubagent,
          cwd: payload.cwd || "",
          transcriptPath,
          transcriptMtimeMs: txStatMtimeMs,
          transcriptSize: txStatSize,
          contextTokens: diag?.currentContextTokens || 0,
          mediaCount: diag?.totalMediaCount || 0,
          model: diag?.currentModel || "",
          cacheTtlSec: diag?.cacheTtlSec || 0,
          ttlSource: diag?.ttlSource || ""
        });
      } catch {}
    }

    // Inherit parent pause state for subagents, while isolating alert suppression keys per direct conversation
    if (!effectiveCfg.cacheWatch || store.isPaused(sessionId) || (isSubagentConv && store.isPaused(targetConvId)) || !diag) {
      store.close();
      store = null;
      process.stdout.write("{}\n");
      return;
    }

    const uiLang =
      typeof store.getUiLang === "function" ? store.getUiLang() : effectiveCfg.uiLang || "en";
    const alerts = evaluateCacheAlerts(diag, {
      sessionId: targetConvId,
      store,
      invocationNum: payload?.invocationNum ?? payload?.invocation_num ?? null,
      lang: uiLang
    });
    store.close();
    store = null;

    // CLI 1.5.14 may panic on metadata.user_markdown (upstream #1394).
    // Keep diagnostics and alert accounting, but do not inject model-visible text as a fallback.
    if (alerts.length === 0 || effectiveCfg.uiOnlyAlerts === false) {
      process.stdout.write("{}\n");
      return;
    }

    const prefixBadge = uiLang === "zh" ? "[agy-dart 缓存守卫]" : "[agy-dart KV-Cache Watch]";
    const markdown = alerts.map((a) => `> **${prefixBadge}** ${a}`).join("\n");
    // Zero-prefix-pollution UI-only step: systemMessage is "" so the model receives 0 tokens!
    process.stdout.write(
      JSON.stringify({
        injectSteps: [
          {
            systemMessage: {
              systemMessage: "",
              metadata: {
                user_markdown: markdown
              }
            }
          }
        ]
      }) + "\n"
    );
  } catch {
    if (store) {
      try {
        store.close();
      } catch {}
    }
    process.stdout.write("{}\n");
  }
}

const __selfPath = path.resolve(fileURLToPath(import.meta.url));
function isSameFilePath(a, b) {
  if (!a || !b) return false;
  const norm = (p) => {
    const resolved = path.resolve(p);
    let real = resolved;
    try {
      real = fs.realpathSync(resolved);
    } catch {}
    return process.platform === "win32" ? real.toLowerCase() : real;
  };
  return norm(a) === norm(b);
}
const isDirectRun = Boolean(process.argv[1] && isSameFilePath(process.argv[1], __selfPath));
if (isDirectRun) {
  main().catch(() => {
    process.stdout.write("{}\n");
  });
}
