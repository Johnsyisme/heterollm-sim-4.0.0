# HBM 几何、控制与成本计量的 JEDEC 依据

`hbm_geometry.py` 直接采用已取得的 JEDEC 协议原文件中的寻址表。文件通过公开镜像取得；镜像站不是标准作者，镜像中的介绍文章和厂商白皮书不作为几何依据。`standard_organization` 只表示采用了标准允许的组织，不表示某张 GPU 或某个厂商料号已被验证。

## 已核对的协议

| 模型名称 | 原文件与直接依据 | 本地文件 | 原文件公开镜像 |
|---|---|---|---|
| HBM2 / HBM2E | JESD235B，2018-11；§3.2.1.2，印刷 p.4；表 4，p.6–7；表 5，p.8 | `.refs/jedec/JESD235B-2018.pdf`，207 PDF 页 | [GitCode 中的 JESD235B-HBM.pdf](https://gitcode.com/Open-source-documentation-tutorial/a1451) |
| HBM3 / HBM3E | JESD238，2022-01；§3.1.2，印刷 p.4–5；表 4/5，p.6–7 | `.refs/jedec/JESD238-2022.pdf`，268 PDF 页 | [GitCode 中的 JESD238-HBM3.pdf](https://gitcode.com/Open-source-documentation-tutorial/69c72) |
| HBM4 | JESD270-4A，2025；§3.1.2；表 4/5，印刷 p.6–7 | `.refs/jedec/JESD270-4A-2025.pdf`，286 PDF 页 | [GitHub 中的 JEDECJESD270-4A-2025.pdf](https://github.com/Adilohit/HBM4_verification/blob/main/resources/JEDECJESD270-4A-2025.pdf) |

另外取得了 [JESD235A 原文件的 Internet Archive 镜像](https://web.archive.org/web/20220514151205id_/https://composter.com.ua/documents/JESD235A.pdf)，存于 `.refs/jedec/JESD235A_archive_composter.pdf`（172 PDF 页），用于历史对照。JESD235C 全文网页镜像、JESD235D 的引用材料、JESD238B.01 和 JESD270-4 的三页预览，不代表取得或核准了这些版本的完整协议；当前实现不使用它们作为几何来源。HBM2E / HBM3E 是模型的技术代名称，不能据此推断其速度等级或厂商实现。

原协议全文只保存在本地 `.refs/jedec`，不随源代码提交。

## 字段口径

| 字段 | HBM2 / HBM2E | HBM3 / HBM3E | HBM4 |
|---|---:|---:|---:|
| 一个完整 stack 的物理通道数 | 8 | 16 | 32 |
| 每物理通道的伪通道数（PC） | 2 | 2 | 2 |
| 每 PC 数据位宽 | 64 bit | 32 bit | 32 bit |
| 每 stack 外部数据总位宽 | 1024 bit | 1024 bit | 2048 bit |
| 突发长度 | BL4 | BL8 | BL8 |
| 每 PC 每次突发数据 | 32 B | 32 B | 32 B |
| 每 PC 有效寻址页 `row_bytes` | 1024 B | 1024 B | 1024 B |
| 每个已激活页 | 当前不作物理大小断言 | 2048 B | 2048 B |
| 每 bank group 的 bank 数 | 4（已导入组织） | 4 | 8 |

PC 是伪通道。每个 PC 有独立可寻址 bank 空间，两个 PC 仍共享所在物理通道的命令接口。`channels` 表示每 stack 的物理通道数；执行模型的 lane 数是 `stacks × channels × 2`。SID 是扩展 bank 地址的 stack ID，不是另一个应再次乘入容量的完整地址空间。因此采用 `bank_geometry_scope="per_stack"`，物理 `dies_per_stack` 只作组织描述。

当前 HBM2/E 目录明确采用 Pseudo Channel mode。JESD235B §3.2.1 同时定义 Legacy mode（称为 HBM1）与 PC mode（称为 HBM2），因此每通道 2 PC 不是该协议覆盖的全部组织。模式支持由器件设计固定：DEVICE_ID[17:16] 的 01 表示只支持 PC，10 表示只支持 Legacy，00/11 非法（表 99，印刷 p.184）；不能推断一个 HBM2 器件可在两者间运行时切换。当前不导入 Legacy 组织。JESD238 §3.1.2 与 JESD270-4A §3.1.2 则直接规定每通道两个 PC，其通道定义没有 Legacy 可选模式；HBM3/E、HBM4 的 2 PC 来自各自协议。

JESD238 和 JESD270-4A 表 4 注 3 明确区分 1 KiB 有效寻址页与已打开的 2 KiB 页：RA 最高位选择其一半。`rows_per_bank` 在本模型中是有效行地址数；`activated_page_bytes=2048` 单独记录激活页大小。协议控制中的已打开行身份须清除该 RA 最高位，两个半页可以共享一次激活。24 Gbit 组织不是 2 的幂：HBM3 清除 RA14，HBM4 清除 RA13；不能使用 `row % (rows_per_bank / 2)` 代替实际地址位掩码。

## 已导入组织与容量

令 D 为单 die 密度（Gbit），H 为每 stack 的物理 die 层数，S 为 stack 数：

| 模型 | 已导入 D | 已导入 H | 每 PC 的 bank 数 | 每 PC 的 bank group 数 | 每 bank 的有效行地址数 |
|---|---|---|---:|---:|---:|
| HBM2 / HBM2E | 16 | 4、8 | 4H | H | 32768 |
| HBM3 / HBM3E | 8、16、24、32 | 4、8、12、16 | 4H | H | D × 1024 |
| HBM4 | 24、32 | 4、8、12、16 | 4H | H / 2 | D × 512 |

HBM2/E 只导入 JESD235B 中的一个有限子集；未导入不等于 JEDEC 禁止。HBM2/E 与 HBM3/E 的无 SID 基础组织，其物理 stack 高度由厂商决定；4H 是本模型对基础组织的具体实现选择。HBM4 表 4 直接列出 4H，注 8 排除 SID[0]=1，不沿用前两代的这一限制。

HBM3 的 24 Gbit 组织排除 RA[14:13]=11，故有效行数为 24576；HBM4 的 24 Gbit 组织排除 RA[13:12]=11，故有效行数为 12288。12H 排除 SID[1:0]=11，得到 3 个 SID × 16 bank = 48 bank/PC。JESD238 原版表 4 的部分脚注上标编号错位；这里按脚注文字、寻址位及表中每 PC 容量核对，未照抄错误上标。

原始可寻址容量使用二进制密度：

```text
raw_capacity_bytes = S × D × H × 2^30 / 8
                   = lane_count × banks_per_PC × rows_per_bank × row_bytes
```

例如 24 Gbit × 12H 的单 stack 是 36 GiB，即 38,654,705,664 B。这个数是寻址组织导出的原始容量，不是对厂商营销标注的推断。`capacity_bytes` 是另行给定的可见容量预算；它可以小于原始容量，但不会通过增大行数去凑容量。多 stack 的带宽输入也按整体预算处理，不再次乘 stack 数。

缺少 D/H 时，选覆盖可见预算的最小原始容量；容量相同时优先 8H、12H、16H、4H。选出的 D/H 标成 `selected_reference`，目标器件的实际 D/H 仍列在 `unknown_parameters`。

## 命令接口与时钟

下列依据均为上表中的完整协议原文件；页码是印刷页码。控制语义、标准允许的编码范围、参考工作点和目标器件保证值是不同层次，不能因几何来自 JEDEC 就把全部模拟时序标成已验证。

| 内容 | JESD235B，HBM2 PC mode | JESD238，HBM3 | JESD270-4A，HBM4 |
|---|---|---|---|
| PC 与命令接口 | §3.2.1.2，p.4；§3.2.2，p.5 | §3.1.2，p.4–5；§3.1.3，p.5 | §3.1.2，p.5；§3.1.3，p.6 |
| CK 与数据速率 | §6.1，p.22；表 67，p.107：每 CK 2 次数据传输 | §6.1，p.26：DQS 为 CK 的 2 倍频，每 CK 4 次数据传输 | §6.1，p.33：每 CK 4 次数据传输 |
| ACT 行命令占用 | 2 CK，表 30，p.30 | 1.5 CK，表 30，p.35 | 1.5 CK，表 33，p.42 |
| PRE / 全 bank REF 行命令占用 | 1 CK | 0.5 CK | 0.5 CK |
| READ / WRITE 列命令占用 | 1 CK，表 31，p.31 | 1 CK，表 31，p.36 | 1 CK，表 34，p.43 |
| 每 PC 一个 32 B 突发的数据占用 | BL4 / 2 = 2 CK | BL8 / 4 = 2 CK | BL8 / 4 = 2 CK |

每个物理通道有分别工作的行命令口与列命令口，两者可以并行；两个 PC 共享这两个物理命令口，各自拥有数据位段与阵列状态。因此不能为每个 PC 配一个完全独立的命令口，也不能用一条串行命令队列把所有行、列操作互相阻塞。数据突发间隔为 2 CK，不是一个命令时钟周期。由带宽推导 CK 时，须注明采用理想满载数据率的假设；读写预算不同、协议开销或降频时，单凭预算不能查得真实 CK。

ACT 必须在上升沿开始，阵列时序从 ACT 的第二个上升沿起算（B §6.3.2.2，p.33；238 同节 p.38；4A 同节 p.46）。HBM3/4 的 ACT 结束后的那个下降沿不是通用发令位置：只允许 RNOP、其他 bank 的 PREpb，或另一 PC 的 PREab；不能把 1.5 CK 占用理解为下一 ACT 可立即从下降沿开始。普通列命令也须满足自己的发令边沿与时序。

## 阵列与数据时序的计量

三代的 PC 时序表均把 tRC、tRAS、tRCDRD/WR、tRRDL/S、tFAW、tRTP、tRP、tWR、tCCDL/S/R、tWTRL/S、tRTW 和刷新时序列为每个 PC 的约束（B 表 3，p.5；238 表 3，p.5；4A 表 3，p.6）。共享命令口的占用仍在物理通道层计量。SID 扩展 bank 地址，不能不加区分地把全部 PC 的 bank 时序合在一起，也不能把每个 SID 当成独立的物理通道。

| 约束 | 起点、终点与作用范围 |
|---|---|
| tRCDRD / tRCDWR | ACT 阵列起点到同 bank READ / WRITE |
| tRAS / tRP / tRC | 激活到预充电、预充电到下一激活、同 bank 两次激活；时序须同时满足 |
| tRRDS / tRRDL | 同 PC 不同 bank 的连续 ACT，分别对应不同 / 相同 bank group |
| tFAW | 同 PC 滚动窗口最多 4 次激活；支持 REFpb 时，该操作也参与相应限制。不能只限制相邻两次 ACT |
| tCCDS / tCCDL | 同 PC 连续同方向列操作，分别对应不同 / 相同 bank group |
| tCCDR | 同 PC 跨 SID 的连续 READ；跨 SID WRITE 仍使用通常的 tCCDS |
| tRTP | READ 命令到同 bank PRE；不是从读数据传输结束起算 |
| tWR | 写数据结束后的恢复时间，不能直接当成 WRITE 命令到 PRE 的全部间隔 |
| tWTRS / tWTRL | 内部写入到 READ 的间隔，分别对应不同 / 相同 bank group；须加上命令到写数据的延迟与突发长度 |

发令间隔依据 B 表 33（p.38）、238 表 33（p.44）、4A 表 36（p.52）。令无量纲 `B = 2` 为本模型各代 PC 突发占用的 CK 周期数，则不含额外 PHY 偏移的基本关系为：

```text
WRITE -> PRE（同 bank） >= WL × tCK + B × tCK + tWR
WRITE -> READ          >= WL × tCK + B × tCK + tWTR
READ  -> PRE（同 bank） >= tRTP
```

公式中的参数必须先按标准表的单位换算：例如 AC 表中的 tRTP、tWTR 是 nCK，而 tWR 是 ns。若实现以 ns 储存，须在配置导入时转换，不能把数字直接相加。READ / WRITE 与另一 bank 的 PRE 可以同时发出，完成时间应遵循资源与依赖约束，而不是把所有开销顺序相加。

HBM3/4 允许 tRAS、tRTP、tWR、tRP 按 `0.5 × ceil(2 × tXX / tCK)` 量化；这一规则只适用于列出的四个参数。PRE 可在下降沿发出，但其后的 ACT 等行访问仍须对齐上升沿（238 p.45；4A p.53）。B 的时序不能直接套用这条半周期规则。

正常 READ -> WRITE 的 tRTW 还包含 PHY 收发偏移，是系统总线换向要求。B 表 68 注 23（p.112）、238 表 93 注 18（p.164）、4A 表 108 注 18（p.198）给出的公式不同：

```text
B:   (RL + BL/2 - WL + tDQSS(min) + 0.5) × tCK
     + tDQSCK(max) + tDQSQ(max)
238: (RL + BL/4 - WL + tDQSS(min) + 0.5) × tCK
     + tWDQS2DQ_O(max) + tDQSQtra(max)
4A:  (RL + BL/4 - WL + 0.5) × tCK
     + tWDQS2DQ_O(max) - tWDQS2DQ_I(min)
```

结果须向上取整为可发令周期。缺少这些 PHY 参数时，可使用明确标注的换向参考值，但不能称为完整执行了上式。

## 数值的来源层次

正式 AC 表是 B 表 68（p.109–112）、238 表 93（p.161–165）、4A 表 108（p.194–199）。其中 tRC、tRAS 最小值、tRCDRD/WR、tRRD、tFAW、tRP、tWR、tRTP、tWTR 等多项最小值在这些原文件中留空；已通过渲染检查表格，不是文本抽取丢失。留空不表示零，不支持声称一个任意 ns 默认值是 JEDEC 器件保证值。

有明确数值的约束包括：三代 PC 模式 tCCDS 最小 2 CK；B 的 tCCDL 最小 `max(4 CK, 2.8 ns)`，238/4A 最小 `max(4 CK, 2.5 ns)`；tRAS 最大值均为 `9 × tREFI`。跨 SID READ 的 tCCDR 是厂商参数：B 给出的支持范围为 2–4 CK（p.112 注 22）；238/4A 为 `tCCDS + 1` 至 `tCCDS + 2` CK（p.164 / p.198 注 17），不能把这个范围误读为总共 1–2 CK。

可以明确命名为 **JEDEC IDD 测量参考工作点，非器件 AC 保证** 的一组参数为 tRC = 48 ns、tRAS = 33 ns、tRP = 15 ns。来源是 B 表 58（p.102）、238 表 84（p.152）、4A 表 99（p.180）；后两代的表标题直接标为 Example。三代都允许厂家使用其他 tRAS / tRP，并要求 `nRAS + nRP = nRC`。按 CK 量化后应保持这项关系，不能假定分别向上取整仍能自动相等。这些表没有提供缺失的 tRCD、tRRD、tFAW、tWTR、tWR 默认值。

RL / WL 是模式寄存器中的时钟周期编码，不是任意的 ns 延迟：

| 协议与出处 | RL 可编码范围 | WL 可编码范围 |
|---|---:|---:|
| B，表 11，p.17（含扩展位） | 3–48 CK | 1–16 CK |
| 238，表 12/13，p.18–19 | 4–63 CK | 4–16 CK |
| 4A，表 11/12，p.21–22 | 17–90 CK | 4–19 CK |

器件不必支持全部编码，实际支持连续的子区间且与速度等级相关。已有 35 ns 分析延迟不能直接声称为 JEDEC RL/WL，高速下转换的周期数甚至可能超出可编码范围。未模拟奇偶校验延迟、训练与 PHY 偏移时，也须保留这些限制。

协议执行路径默认选择 RL = 32 CK、WL = 8 CK，作为三代编码范围内的 `selected_reference`。配置中的读写延迟据此换算为 ns，并校验为整 CK 且处于本代编码范围；这不证明目标器件支持这些编码。tRAS / tRP 使用上述 33 / 15 ns IDD 参考，tRC 保持量化后的 `nRAS + nRP`。tRCD = 14 ns、tWR = 15 ns 仍是分析假设；未取得数值的 tRRD、tFAW、tWTR 为 0 时只表示该约束未建模，不表示标准允许零间隔。tCCDR 选择范围上限 4 CK，也属于未验证厂商参数的参考选择。数据总线读写避冲突可以防止传输重叠，但没有完整 PHY 参数时仍不保证满足实际器件的 tRTW。

## 刷新

全 bank REF 命令选择一个 PC，不选择 SID；一次作用于该 PC 的所有 SID、所有 bank。刷新前须使相应 bank 空闲并满足 tRP，刷新后该 PC 在 tRFC 内不可进行阵列访问；另一 PC 仍有自身状态。依据 B 表 30 / §6.3.2.4（p.30 / p.39）、238 表 30 / §6.3.2.4（p.35 / p.46–47）、4A 表 33 / §6.3.2.4（p.42 / p.55–56）。HBM3/4 刷新后的半周期仍有限定命令组合，不能将其当成任意空槽。

tREFI 是平均刷新间隔，正常温度基础表值为最大 3.9 μs；温度条件可能要求减半或四分之一。HBM3/4 允许受限延期与提前刷新（最多延期 8 次、绝对间隔不超过 9 × tREFI），周期性不延期策略是其一个简单子集。仅记录刷新命令次数而不计预充电及 tRFC 阻塞，会低估时间成本。

以下 tRFC 为原文件 AC 表的全 bank 刷新值，单位 ns；按 die 密度与堆叠层数查表，不能把每通道密度错当成单 die 密度：

| 组织 | 4H | 8H | 12H | 16H | 原文件位置 |
|---|---:|---:|---:|---:|---|
| B，16 Gbit/die（当前导入子集） | 350 | 450 | 未导入 | 未导入 | 表 68，p.110；对应 8 / 16 Gbit 每通道 |
| 238，8 Gbit/die | 未给出 | 260 | 310 | 350 | 表 93，p.162 |
| 238，16 Gbit/die | 260 | 350 | 410 | 450 | 表 93，p.162 |
| 238，24 / 32 Gbit/die | TBD | TBD | TBD | TBD | 表 93，p.162 |
| 4A，24 Gbit/die | 360 | 410 | 450 | 490 | 表 108，p.195 |
| 4A，32 Gbit/die | 400 | 450 | 490 | 530 | 表 108，p.195 |

TBD 不等于零。缺少有效 tRFC 时，刷新模型应明确禁用并列为未模拟，或者要求显式参考值；不能把其他技术代的数字标成这一代的 JEDEC 数值。REFpb 还需要逐 bank / SID 的刷新覆盖计数与 tRREFD、tRFCpb、tFAW 等约束，不能用全 bank REF 简单拆分来声称完整支持。

默认参考策略仅对上表有确定 tRFC 的已导入组织启用每 3900 ns 一次的周期性全 bank 刷新；HBM3 24 / 32 Gbit 及其他 TBD 组织明确禁用。该策略的温度条件尚未验证，不能用于声称覆盖所有工作温度。

最大激活时间与刷新开关分别配置。默认 `maximum_active_ns = 9 × 3900 = 35100 ns`，包括因 tRFC 未知而禁用刷新的组织；发令前检查行的关闭期限与本次读写恢复时间，必要时先在期限内预充电再激活。排不出合法关闭位置时明确报错。原始自定义配置中的 0 表示未配置这条约束，不表示标准没有最大值。

## 成本口径与尚未确定的内容

协议能够确定命令数、命令口占用、传输字节、阵列依赖与等待时间。32 B 是每 PC 一个突发的数据量，不能按请求字节数无条件计一个 READ / WRITE，也不能将有效页 1 KiB 计成一次数据突发。并行 bank、PC、物理通道的操作可以重叠：延迟取依赖路径与共享资源排程的完成时间；各资源 busy time 的和是另一项统计量，不应当作墙钟耗时。

汇总执行使用独立的 HBM float64 编译路径，仍执行这套命令、SID、行状态和刷新逻辑，并回写与明细路径相同的状态与资源计数。只在 bank 全部关闭、命令口与 PC 完全空闲、且浮点运算可证明精确的时间网格及数值范围内折叠空闲刷新周期；被折叠的刷新次数和占用仍全部计入。需要明细、非零 tFAW、实际合并或重命名本 HBM 资源等未加速情形回到同一套 HBM 明细控制，不切回旧 DRAM 控制模型。内核中无关资源的别名不影响 HBM 加速。

`jedec_hbm` 执行使用 `activate_to_read_ns` / `activate_to_write_ns`、`precharge_to_activate_ns`、`write_to_precharge_ns` 等新时序字段。通过 `reconfigure_physical_config` 修改旧 `open_ns` / `close_ns` / `write_recovery_ns` 时，会同步对应新字段；同时显式给出的新字段优先。直接读取已序列化配置时，旧 `open_ns` / `close_ns` / `burst_interval_ns` / `read_recovery_ns` / `write_recovery_ns` 只作兼容保存，不直接参与新执行路径。RL/WL 的 ns 配置必须已经对应整 CK，校验不会自动取整改变输入。

JEDEC 的电流测量条件与模式在各协议 §9.1（B p.100 起，238 p.149 起，4A p.177 起）。IDD / IPP 等是指定工作模式、供电轨与负载下的电流条件，实际电流数值需要器件数据；标准不提供统一的 pJ/READ、pJ/ACT 或货币单价。若模拟能量，应另给电压、电流及背景电流扣除方式，并按状态驻留时间计算，避免背景能量和命令能量重复计入；已有每字节或每命令能耗常数属于单独的分析模型。

厂商料号、厂商地址映射、实际速度等级、支持的 RL/WL 子区间、未填阵列时序、PHY 换向极值、温度刷新条件、电流与能量参数仍未由这些寻址表和协议参考工作点确定。初始化、训练、低功耗、奇偶校验、故障恢复及可选刷新机制也不因支持普通读写而自动得到实现。配置的逐字段来源和未知项保存在 `metadata.parameter_provenance` 与 `metadata.unknown_parameters` 中；协议结构正确与某个真实器件的时序已验证必须分别表达。
