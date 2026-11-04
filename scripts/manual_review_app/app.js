"use strict";
(() => {
  const D = window.REVIEW_DATA;
  const $ = id => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  const KEY = `flood-manual-review:${D.dataset_id}`;
  const GROUPS = {all:"全部",false_positive:"误检提示",conflict:"等级冲突",candidates:"其他候选",no_detection:"未检出"};
  const STATUSES = {pending:"待复核",done:"已复核",deferred:"暂存疑问"};
  const VERDICTS = {pending:"待判断",keep:"目标正确",reject:"误检",uncertain:"不确定"};
  const TYPES = {car:"汽车",person:"人物",motorcycle:"两轮车",other:"其他"};
  const POSITIONS = {
    car:{pending:"尚未判断",uncertain:"水线无法判断",no_submergence:"未见明显淹没",lower_tire:"水至轮胎下部",upper_tire:"水至轮胎上部 / 接近轮拱",body:"水至车身",window_or_above:"水至车窗及以上"},
    person:{pending:"尚未判断",uncertain:"水线无法判断",no_submergence:"未见明显淹没",ankle:"水至脚踝附近",shin:"水至小腿",knee:"水至膝部附近",thigh:"水至大腿",waist:"水至腰部附近",chest_or_above:"水至胸部及以上"},
    motorcycle:{pending:"尚未判断",uncertain:"水线无法判断",no_submergence:"未见明显淹没",lower_wheel:"水至车轮下部",upper_wheel:"水至车轮上部",body:"水至车身"},
    other:{pending:"尚未判断",uncertain:"水线无法判断",no_submergence:"未见明显淹没",partial:"局部淹没，见备注",mostly:"大部分淹没，见备注"}
  };
  const byId = new Map(D.images.map(r => [r.image_id,r]));
  const ranks = {false_positive:0,conflict:1,candidates:2,no_detection:3};
  const ordered = [...D.images].sort((a,b)=>ranks[a.group]-ranks[b.group] || a.ordinal-b.ordinal);
  let state = {schema:"flood_manual_review_v1",dataset_id:D.dataset_id,reviews:{}};
  let storageOK = true;
  let currentId = ordered[0].image_id, selectedId = null, group = "all", zoom = 1, fitScale = 1;
  let showBoxes = true, drawing = false, dragStart = null, dragRect = null, toastTimer;
  let filtered = [];
  const defaultImage = () => ({status:"pending",scene_water:"pending",reference_availability:"pending",note:"",targets:{},manual_targets:[],updated_at:null});
  const defaultTarget = type => ({verdict:"pending",reference_type:type||"car",level_verdict:"pending",manual_level:null,waterline:"pending",note:""});
  const row = () => byId.get(currentId);
  const imageReview = id => state.reviews[id] || defaultImage();
  const activeReview = () => imageReview(currentId);
  const allTargets = r => [...r.detections.map((t,i)=>({...t,id:t.detection_id,origin:"model",number:i+1})),...(imageReview(r.image_id).manual_targets||[]).map((t,i)=>({...t,origin:"manual",number:r.detections.length+i+1}))];
  const selected = () => allTargets(row()).find(t=>t.id===selectedId);
  const targetReview = t => activeReview().targets[t.id] || defaultTarget(t.reference_type);
  function toast(message) { $("toast").textContent=message; $("toast").style.display="block"; clearTimeout(toastTimer); toastTimer=setTimeout(()=>$("toast").style.display="none",4200); }
  function persist() {
    try { localStorage.setItem(KEY,JSON.stringify(state)); storageOK=true; $("saveState").textContent="已在本浏览器暂存 · 请定期导出 JSON"; }
    catch { storageOK=false; $("saveState").textContent="浏览器暂存不可用 · 请导出 JSON 保存"; }
  }
  function edit(mutator) {
    if (!state.reviews[currentId]) state.reviews[currentId]=defaultImage();
    const r=state.reviews[currentId];
    if(r.status==="done") r.status="pending";
    mutator(r); r.updated_at=new Date().toISOString(); persist(); updateProgress(); renderList(); updateImageStatus();
  }
  function editTarget(mutator) {
    const t=selected(); if(!t)return;
    edit(r=>{ if(!r.targets[t.id])r.targets[t.id]=defaultTarget(t.reference_type); mutator(r.targets[t.id]); });
    renderTargets(); renderBoxes();
  }
  function updateProgress() {
    const n=D.images.filter(r=>imageReview(r.image_id).status==="done").length;
    $("progressText").textContent=`已完成 ${n} / ${D.image_count}`; $("progress").max=D.image_count; $("progress").value=n;
  }
  function updateImageStatus() { const s=activeReview().status; $("imageStatus").className="status "+s; $("imageStatus").textContent=STATUSES[s]; }
  function renderFilters() {
    $("groupFilters").replaceChildren();
    for(const [key,label] of Object.entries(GROUPS)) {
      const b=document.createElement("button"); b.textContent=`${label} ${key==="all"?D.image_count:D.group_counts[key]||0}`; b.className=key===group?"active":""; b.onclick=()=>{group=key;renderFilters();renderList();if(filtered.length&&!filtered.some(r=>r.image_id===currentId))openImage(filtered[0].image_id);}; $("groupFilters").append(b);
    }
  }
  function renderList() {
    const previousScroll=$("photoList").scrollTop;
    const query=$("search").value.trim().toLowerCase(), status=$("reviewFilter").value;
    filtered=ordered.filter(r=>(group==="all"||r.group===group) && (!query||`${String(r.ordinal).padStart(3,"0")} ${r.image_id} ${r.location}`.toLowerCase().includes(query)) && (status==="all" || (status==="pending"?imageReview(r.image_id).status!=="done":imageReview(r.image_id).status===status)));
    $("listCount").textContent=`当前列表 ${filtered.length} 张`; $("photoList").replaceChildren();
    for(const r of filtered){
      const b=document.createElement("button"), s=imageReview(r.image_id).status;
      b.className="photo-row"+(r.image_id===currentId?" active":""); b.dataset.imageId=r.image_id;
      const top=document.createElement("span");top.className="row-top";top.textContent=`${String(r.ordinal).padStart(3,"0")} · ${GROUPS[r.group]}`;
      const statusNode=document.createElement("span");statusNode.className="dot "+s;statusNode.title=STATUSES[s];top.append(statusNode);
      const name=document.createElement("span");name.className="row-title";name.textContent=r.image_id.replace(/__[0-9a-f]{10}$/i,"");
      const meta=document.createElement("span");meta.className="row-meta";meta.textContent=`${r.detections.length} 个模型框 · ${STATUSES[s]}`;
      b.append(top,name,meta);b.onclick=()=>openImage(r.image_id);$("photoList").append(b);
    }
    $("photoList").scrollTop=previousScroll;
    $("positionText").textContent=`${Math.max(0,filtered.findIndex(r=>r.image_id===currentId)+1)} / ${filtered.length}`;
  }
  function openImage(id) {
    currentId=id; selectedId=null;zoom=1;drawing=false;$("imageViewport").classList.remove("drawing");$("drawButton").textContent="＋ 补框";
    const r=row(), review=activeReview();selectedId=allTargets(r)[0]?.id||null;
    $("photoOrdinal").textContent=`照片 ${String(r.ordinal).padStart(3,"0")} / ${D.image_count} · ${GROUPS[r.group]}`;
    $("photoName").textContent=r.image_id;$("photoContext").textContent=[r.location,r.observed_time].filter(v=>v&&!/^(unknown|none|null)$/i.test(v)).map(v=>String(v).replace(/\bunknown\b/gi,"未核实")).join(" · ")||"地点和拍摄时间未核实";
    $("originalImage").src=r.original;$("originalLink").href=r.original;$("originalImage").alt=`照片${r.ordinal}原图`;
    $("boxLayer").setAttribute("viewBox",`0 0 ${r.width} ${r.height}`);
    $("sceneWater").value=review.scene_water;$("referenceAvailability").value=review.reference_availability;$("imageNote").value=review.note;
    $("advisory").replaceChildren();const label=document.createElement("strong");label.textContent="已有提示（供参考）： ";$("advisory").append(label,document.createTextNode(r.advisory_note|| (r.detections.length?"尚无逐目标复核结论。请先确认框内是否为车辆，再判断可见水线。":"模型未检出目标。请查看是否有遗漏的人物或其他参照物；不能据此记为零水深。")));
    $("advisory").className="advisory"+(!r.advisory_note?" neutral":"");
    if(r.overlap_flags.length){const p=document.createElement("div");p.textContent="⚑ 存在不同等级的重叠框，注意重复检测或目标遮挡。";$("advisory").append(p);}
    renderEvidence();renderTargets();renderEditor();renderList();updateImageStatus();
    const activeButton=$("photoList").querySelector(".active"),list=$("photoList");
    if(activeButton){const offset=activeButton.offsetTop-list.offsetTop;if(offset<list.scrollTop)list.scrollTop=offset;else if(offset+activeButton.offsetHeight>list.scrollTop+list.clientHeight)list.scrollTop=offset+activeButton.offsetHeight-list.clientHeight;}
    requestAnimationFrame(fitImage);
  }
  function fitImage() {
    const v=$("imageViewport"),r=row();fitScale=Math.min((v.clientWidth-4)/r.width,(v.clientHeight-4)/r.height);
    $("imageStage").style.width=`${r.width*fitScale*zoom}px`;$("imageStage").style.height=`${r.height*fitScale*zoom}px`;
    $("zoomValue").textContent=`${Math.round(zoom*100)}%`;renderBoxes();
  }
  function renderBoxes() {
    const svg=$("boxLayer");svg.replaceChildren();if(!showBoxes&&!drawing)return;
    for(const t of allTargets(row())) {
      const a=targetReview(t), box=t.bbox_xyxy, g=document.createElementNS(NS,"g"),rect=document.createElementNS(NS,"rect");
      const color=a.verdict==="reject"?"#bd4a35":t.id===selectedId?"#ffd65b":t.origin==="manual"?"#9b7de0":"#49baaf";
      rect.setAttribute("x",box[0]);rect.setAttribute("y",box[1]);rect.setAttribute("width",box[2]-box[0]);rect.setAttribute("height",box[3]-box[1]);rect.setAttribute("stroke",color);rect.setAttribute("stroke-width",t.id===selectedId?3:1.7);rect.classList.toggle("selected-box",t.id===selectedId);
      if(t.origin==="manual")rect.setAttribute("stroke-dasharray","6 3");
      const text=document.createElementNS(NS,"text");text.setAttribute("x",box[0]+3/(fitScale*zoom));text.setAttribute("y",Math.max(16/(fitScale*zoom),box[1]-4/(fitScale*zoom)));text.setAttribute("font-size",12/(fitScale*zoom));text.textContent=`#${t.number} ${t.origin==="model"?"L"+t.level:"补框"}`;
      g.append(rect,text);g.onclick=e=>{if(drawing)return;e.stopPropagation();selectedId=t.id;renderTargets();renderEditor();renderBoxes();};svg.append(g);
    }
  }
  function renderTargets() {
    const list=$("targetList");list.replaceChildren();const targets=allTargets(row());
    if(!targets.length){const empty=document.createElement("p");empty.className="empty-state";empty.textContent="没有模型框。发现参照物时可以补框；没有可用参照物也可以如实记录。";list.append(empty);}
    for(const t of targets){const b=document.createElement("button");b.className="target-row"+(t.id===selectedId?" active":"");b.dataset.targetId=t.id;const left=document.createElement("span"),right=document.createElement("span");left.textContent=t.origin==="model"?`#${t.number} · L${t.level} · ${(t.confidence*100).toFixed(0)}%`:`#${t.number} · 人工补框`;right.className="target-state";right.textContent=VERDICTS[targetReview(t).verdict];b.append(left,right);b.onclick=()=>{selectedId=t.id;renderTargets();renderEditor();renderBoxes();};list.append(b);}
    $("rejectAll").disabled=!row().detections.length;
  }
  function renderEditor() {
    const t=selected();$("targetEditor").hidden=!t;if(!t)return;
    const a=targetReview(t);$("selectedTargetTitle").textContent=`正在复核目标 #${t.number}${t.origin==="model"?` · 模型 L${t.level}`:" · 人工补框"}`;
    for(const [key,v] of [["keepTarget","keep"],["rejectTarget","reject"],["unsureTarget","uncertain"]])$(key).classList.toggle("chosen",a.verdict===v);
    $("referenceType").value=a.reference_type;$("levelVerdict").value=a.level_verdict;$("manualLevel").value=a.manual_level===null?"":String(a.manual_level);
    $("manualLevel").disabled=t.origin==="manual"||a.level_verdict!=="disagree";
    $("levelVerdict").disabled=t.origin==="manual"||a.verdict==="reject"||a.reference_type!=="car";
    $("waterline").replaceChildren();for(const [value,text] of Object.entries(POSITIONS[a.reference_type])){const o=document.createElement("option");o.value=value;o.textContent=text;$("waterline").append(o);}$("waterline").value=a.waterline;
    $("targetNote").value=a.note;$("removeManual").hidden=t.origin!=="manual";
  }
  function renderEvidence() {
    const r=row(),c=$("evidenceContent");c.replaceChildren();$("evidencePanel").open=false;$("evidenceCount").textContent=r.evidence.length?`· ${r.evidence.length} 张诊断图`:"";
    const p=document.createElement("p");p.textContent="下列试算不是现场真值。人物身高比例、轮毂直径比例与模型等级不能直接相互换算。";c.append(p);
    if(r.prior_wheel_note){const n=document.createElement("p");n.textContent="旧车辆复核："+r.prior_wheel_note;c.append(n);}
    for(const e of r.evidence){const article=document.createElement("div");article.className="evidence-entry";const a=document.createElement("a");a.href=e.image;a.target="_blank";a.rel="noopener";const img=document.createElement("img");img.src=e.image;img.loading="lazy";img.alt="既有参照物试算诊断图";a.append(img);const note=document.createElement("p");note.textContent=`${e.kind==="person"?"人物身高比例":"轮毂直径比例"}试算 ${e.fraction===null?"不可用":(e.fraction*100).toFixed(1)+"%"}（未验证）：${e.note}`;article.append(a,note);c.append(article);}
    for(const [i,url] of r.source_urls.entries()){const a=document.createElement("a");a.href=url;a.target="_blank";a.rel="noopener";a.textContent=`来源网页 ${i+1} ↗ `;c.append(a);}
  }
  function move(delta) {
    renderList();const idx=filtered.findIndex(r=>r.image_id===currentId),next=filtered[idx+delta];
    if(next)openImage(next.image_id);else toast(delta>0?"已到当前列表末尾，可切换筛选继续复核。":"已到当前列表开头。");
  }
  function complete() {
    const r=activeReview();if(r.scene_water==="pending"||r.reference_availability==="pending"){toast("请先填写整图的积水和参照物情况。");return;}
    for(const t of allTargets(row())) {const a=targetReview(t);if(a.verdict==="pending"||(a.verdict==="keep"&&((t.origin==="model"&&a.reference_type==="car"&&a.level_verdict==="pending")||a.waterline==="pending"))){selectedId=t.id;renderTargets();renderEditor();renderBoxes();toast(`目标 #${t.number} 的目标、等级或水线尚未判断；看不清可选择“无法判断”。`);return;}}
    const oldIndex=filtered.findIndex(r=>r.image_id===currentId), nextId=filtered[oldIndex+1]?.image_id;
    edit(r=>r.status="done");if(nextId)openImage(nextId);else toast("本张已完成。当前列表已到末尾，请导出 JSON 备份。");
  }
  function setVerdict(v){editTarget(a=>{a.verdict=v;if(v==="reject") {a.level_verdict="not_applicable";a.manual_level=null;}else if(v==="uncertain")a.level_verdict="uncertain";else if(a.level_verdict==="not_applicable")a.level_verdict=selected().origin==="manual"?"not_applicable":"pending";});renderEditor();}
  function bundle() {
    return {schema:"flood_manual_review_v1",dataset_id:D.dataset_id,exported_at:new Date().toISOString(),
      interpretation:"人工视觉复核记录，不是实测水深。没有检测不能解释为零水深。",images:D.images.map(r=>{
        const a=imageReview(r.image_id);return {image_id:r.image_id,source_sha256:r.source_sha256,filename:r.filename,...a,
          model_detections:r.detections.map(t=>({detection_id:t.detection_id,level:t.level,confidence:t.confidence,bbox_xyxy:t.bbox_xyxy})),
          estimated_depth_cm:null,accuracy_validated:false};})};
  }
  function download(text,type,filename) {const blob=new Blob([text],{type});const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=filename;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),20000);}
  const stamp=()=>new Intl.DateTimeFormat("sv-SE",{timeZone:"Asia/Shanghai",year:"numeric",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false}).format(new Date()).replace(/[ :]/g,"-");
  function exportJson(){download(JSON.stringify(bundle(),null,2),"application/json",`人工复核备份-${stamp()}.json`);toast("已请求下载 JSON 备份，请确认下载文件已保存。");}
  function exportCsv(){
    const rows=[["照片序号","图片ID","图片复核状态","是否可见积水","可用参照物","目标编号","目标来源","模型等级","模型置信度","目标判断","参照物类型","模型等级判断","人工修正等级","相对水线","整图备注","目标备注","更新时间"]];
    const water={pending:"待判断",water:"可见积水",no_visible_water:"未见明显积水",uncertain:"无法判断"},refs={pending:"待判断",available:"有可用参照物",none:"未见可用参照物",uncertain:"不确定"},levels={pending:"待判断",agree:"同意",disagree:"不同意",uncertain:"无法判断",not_applicable:"不适用"};
    for(const r of D.images){const rev=imageReview(r.image_id),ts=allTargets(r);for(const t of ts.length?ts:[null]){const a=t?(rev.targets[t.id]||defaultTarget(t.reference_type)):null;rows.push([r.ordinal,r.image_id,STATUSES[rev.status],water[rev.scene_water],refs[rev.reference_availability],t?.number||"",t?(t.origin==="model"?"模型":"人工补框"):"无目标",t?.level??"",t?.confidence??"",a?VERDICTS[a.verdict]:"",a?TYPES[a.reference_type]:"",a?levels[a.level_verdict]:"",a?.manual_level??"",a?POSITIONS[a.reference_type][a.waterline]:"",rev.note,a?.note||"",rev.updated_at||""]);}}
    const escape=value=>{let s=String(value??"");if(/^[\s]*[=+@-]/.test(s))s="'"+s;return '"'+s.replace(/"/g,'""')+'"';};
    download("\ufeff"+rows.map(r=>r.map(escape).join(",")).join("\r\n"),"text/csv;charset=utf-8",`人工复核表-${stamp()}.csv`);toast("已导出复核 CSV；恢复工作请保留 JSON 备份。");
  }
  function validateReview(id,raw) {
    const source=byId.get(id);if(!source||!raw||typeof raw!=="object")throw Error("记录包含未知照片。");
    const enums=(value,allowed)=>{if(!allowed.includes(value))throw Error("记录中存在无效选项。");return value;};
    const text=value=>{if(typeof value!=="string"||value.length>12000)throw Error("记录文本无效或过长。");return value;};
    const out=defaultImage();out.status=enums(raw.status,["pending","done","deferred"]);out.scene_water=enums(raw.scene_water,["pending","water","no_visible_water","uncertain"]);out.reference_availability=enums(raw.reference_availability,["pending","available","none","uncertain"]);out.note=text(raw.note);
    if(raw.updated_at!==null && (typeof raw.updated_at!=="string"||!Number.isFinite(Date.parse(raw.updated_at))))throw Error("记录时间无效。");out.updated_at=raw.updated_at;
    if(!Array.isArray(raw.manual_targets)||raw.manual_targets.length>100)throw Error("人工补框数据无效。");
    const ids=new Set(source.detections.map(t=>t.detection_id));
    out.manual_targets=raw.manual_targets.map(t=>{if(typeof t.id!=="string"||!t.id.startsWith(id+"__manual_")||ids.has(t.id))throw Error("补框编号无效。");ids.add(t.id);const b=t.bbox_xyxy;if(!Array.isArray(b)||b.length!==4||!b.every(Number.isFinite)||b[0]<0||b[1]<0||b[2]>source.width||b[3]>source.height||b[2]<=b[0]||b[3]<=b[1])throw Error("补框坐标越界。");return{id:t.id,bbox_xyxy:b,reference_type:enums(t.reference_type,Object.keys(TYPES))};});
    if(!raw.targets||typeof raw.targets!=="object"||Array.isArray(raw.targets))throw Error("目标记录无效。");
    for(const [targetId,a] of Object.entries(raw.targets)){if(!ids.has(targetId))throw Error("记录含有未知目标。");const type=enums(a.reference_type,Object.keys(TYPES));const level=a.manual_level;if(level!==null&&(!Number.isInteger(level)||level<0||level>4))throw Error("人工等级无效。");out.targets[targetId]={verdict:enums(a.verdict,Object.keys(VERDICTS)),reference_type:type,level_verdict:enums(a.level_verdict,["pending","agree","disagree","uncertain","not_applicable"]),manual_level:level,waterline:enums(a.waterline,Object.keys(POSITIONS[type])),note:text(a.note)};}
    if(out.status==="done"&&(out.scene_water==="pending"||out.reference_availability==="pending"||[...ids].some(t=>{const a=out.targets[t];return !a||a.verdict==="pending"||(a.verdict==="keep"&&(a.waterline==="pending"||(source.detections.some(d=>d.detection_id===t)&&a.reference_type==="car"&&a.level_verdict==="pending")));})))throw Error("已完成记录仍含待判断项目。");
    return out;
  }
  async function importFile(file) {
    if(!file)return;if(file.size>10000000){toast("文件过大，请选择本页面导出的 JSON。");return;}
    try{const parsed=JSON.parse(await file.text());if(parsed.schema!=="flood_manual_review_v1"||parsed.dataset_id!==D.dataset_id||!Array.isArray(parsed.images))throw Error("这不是同批照片的复核备份，请选择本页面导出的 JSON。");
      const incoming={},seen=new Set();for(const r of parsed.images){if(seen.has(r.image_id)||r.source_sha256!==byId.get(r.image_id)?.source_sha256)throw Error("照片编号重复或来源校验不匹配。");seen.add(r.image_id);incoming[r.image_id]=validateReview(r.image_id,r);}
      if(seen.size!==D.image_count)throw Error("备份照片数量不完整。");
      let merged=0;for(const [id,r] of Object.entries(incoming)){const local=state.reviews[id];if(r.updated_at&&(!local?.updated_at||Date.parse(r.updated_at)>Date.parse(local.updated_at))){state.reviews[id]=r;merged++;}}
      persist();updateProgress();openImage(currentId);toast(`已合并 ${merged} 张较新的记录；未改动原始模型结果。`);
    }catch(error){toast("导入失败："+error.message);}finally{$("importFile").value="";}
  }
  $("search").oninput=renderList;$("reviewFilter").onchange=()=>{renderList();if(filtered.length&&!filtered.some(r=>r.image_id===currentId))openImage(filtered[0].image_id);};
  $("sceneWater").onchange=e=>edit(r=>r.scene_water=e.target.value);$("referenceAvailability").onchange=e=>edit(r=>r.reference_availability=e.target.value);$("imageNote").oninput=e=>edit(r=>r.note=e.target.value);
  $("keepTarget").onclick=()=>setVerdict("keep");$("rejectTarget").onclick=()=>setVerdict("reject");$("unsureTarget").onclick=()=>setVerdict("uncertain");
  $("referenceType").onchange=e=>{const t=selected(),type=e.target.value;editTarget(a=>{a.reference_type=type;a.waterline="pending";if(t.origin==="manual"||type!=="car"){a.level_verdict="not_applicable";a.manual_level=null;}else if(a.verdict!=="reject")a.level_verdict="pending";});if(t.origin==="manual")edit(r=>r.manual_targets.find(m=>m.id===t.id).reference_type=type);renderEditor();};
  $("levelVerdict").onchange=e=>{editTarget(a=>{a.level_verdict=e.target.value;if(a.level_verdict!=="disagree")a.manual_level=null;});renderEditor();};
  $("manualLevel").onchange=e=>editTarget(a=>a.manual_level=e.target.value===""?null:Number(e.target.value));$("waterline").onchange=e=>editTarget(a=>a.waterline=e.target.value);$("targetNote").oninput=e=>editTarget(a=>a.note=e.target.value);
  $("rejectAll").onclick=()=>{edit(r=>{for(const t of row().detections)r.targets[t.detection_id]={...(r.targets[t.detection_id]||defaultTarget()),verdict:"reject",level_verdict:"not_applicable",manual_level:null};});renderTargets();renderEditor();renderBoxes();toast("已标记所有模型框为误检；整图积水和参照物情况仍需填写。");};
  $("previous").onclick=()=>move(-1);$("next").onclick=()=>move(1);$("completeButton").onclick=complete;
  $("deferButton").onclick=()=>{const idx=filtered.findIndex(r=>r.image_id===currentId),nextId=filtered[idx+1]?.image_id;edit(r=>r.status="deferred");if(nextId)openImage(nextId);else toast("已暂存疑问。当前列表已到末尾。");};
  function toggleBoxes(){showBoxes=!showBoxes;$("toggleBoxes").textContent=showBoxes?"隐藏框 B":"显示框 B";$("toggleBoxes").setAttribute("aria-pressed",String(showBoxes));renderBoxes();}
  $("toggleBoxes").onclick=toggleBoxes;$("fitButton").onclick=()=>{zoom=1;fitImage();$("imageViewport").scrollTo(0,0);};
  $("zoomIn").onclick=()=>{zoom=Math.min(8,zoom*1.5);fitImage();};$("zoomOut").onclick=()=>{zoom=Math.max(.5,zoom/1.5);fitImage();};
  $("focusTarget").onclick=()=>{const t=selected();if(!t){toast("请先选择一个目标。");return;}const b=t.bbox_xyxy,v=$("imageViewport");zoom=Math.min(8,Math.max(1,.72*Math.min(v.clientWidth/(b[2]-b[0]),v.clientHeight/(b[3]-b[1]))/fitScale));fitImage();v.scrollLeft=(b[0]+b[2])/2*fitScale*zoom-v.clientWidth/2;v.scrollTop=(b[1]+b[3])/2*fitScale*zoom-v.clientHeight/2;};
  $("drawButton").onclick=()=>{drawing=!drawing;$("imageViewport").classList.toggle("drawing",drawing);$("drawButton").textContent=drawing?"取消补框":"＋ 补框";if(drawing){showBoxes=true;$("toggleBoxes").textContent="隐藏框 B";$("toggleBoxes").setAttribute("aria-pressed","true");toast("请在原图中拖出参照物范围，再在右侧选择类型。");}renderBoxes();};
  const point=e=>{const b=$("imageStage").getBoundingClientRect(),r=row();return[Math.max(0,Math.min(r.width,(e.clientX-b.left)/(fitScale*zoom))),Math.max(0,Math.min(r.height,(e.clientY-b.top)/(fitScale*zoom)))];};
  $("boxLayer").onpointerdown=e=>{if(!drawing||e.button!==0)return;e.preventDefault();dragStart=point(e);$("boxLayer").setPointerCapture(e.pointerId);dragRect=document.createElementNS(NS,"rect");dragRect.setAttribute("stroke","#f0c94e");dragRect.setAttribute("stroke-width","3");$("boxLayer").append(dragRect);};
  $("boxLayer").onpointermove=e=>{if(!dragStart)return;const p=point(e);dragRect.setAttribute("x",Math.min(p[0],dragStart[0]));dragRect.setAttribute("y",Math.min(p[1],dragStart[1]));dragRect.setAttribute("width",Math.abs(p[0]-dragStart[0]));dragRect.setAttribute("height",Math.abs(p[1]-dragStart[1]));};
  $("boxLayer").onpointerup=e=>{if(!dragStart)return;const p=point(e),b=[Math.min(p[0],dragStart[0]),Math.min(p[1],dragStart[1]),Math.max(p[0],dragStart[0]),Math.max(p[1],dragStart[1])];dragStart=null;if((b[2]-b[0])*fitScale*zoom<5||(b[3]-b[1])*fitScale*zoom<5){renderBoxes();toast("补框太小，请重新拖动。");return;}const id=currentId+"__manual_"+(crypto.randomUUID?.()||Date.now().toString(36));edit(r=>{r.manual_targets.push({id,bbox_xyxy:b,reference_type:"car"});r.targets[id]={...defaultTarget(),level_verdict:"not_applicable"};});selectedId=id;drawing=false;$("drawButton").textContent="＋ 补框";$("imageViewport").classList.remove("drawing");renderTargets();renderEditor();renderBoxes();};
  $("boxLayer").onpointercancel=()=>{dragStart=null;renderBoxes();};
  $("removeManual").onclick=()=>{const t=selected();if(!t||t.origin!=="manual")return;edit(r=>{r.manual_targets=r.manual_targets.filter(x=>x.id!==t.id);delete r.targets[t.id];});selectedId=allTargets(row())[0]?.id||null;renderTargets();renderEditor();renderBoxes();};
  $("exportJson").onclick=exportJson;$("exportCsv").onclick=exportCsv;$("importButton").onclick=()=>$("importFile").click();$("importFile").onchange=e=>importFile(e.target.files[0]);$("helpButton").onclick=()=>$("helpDialog").showModal();
  document.addEventListener("keydown",e=>{if($("helpDialog").open||/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)||e.ctrlKey||e.metaKey||e.altKey)return;if(e.key.toLowerCase()==="j"){e.preventDefault();move(1);}else if(e.key.toLowerCase()==="k"){e.preventDefault();move(-1);}else if(e.key.toLowerCase()==="b"){e.preventDefault();toggleBoxes();}else if(e.key==="Enter"){e.preventDefault();complete();}});
  window.addEventListener("resize",fitImage);$("originalImage").onload=fitImage;
  try{const raw=localStorage.getItem(KEY);if(raw){const parsed=JSON.parse(raw);if(parsed.schema!==state.schema||parsed.dataset_id!==D.dataset_id)throw Error("版本不匹配");const restored={};for(const [id,r] of Object.entries(parsed.reviews||{}))restored[id]=validateReview(id,r);state.reviews=restored;$("saveState").textContent="已恢复本浏览器上次暂存 · 请定期导出 JSON";}}
  catch{storageOK=false;$("saveState").textContent="暂存无法恢复 · 请导入已导出的 JSON 备份";}
  renderFilters();updateProgress();openImage(currentId);
})();
