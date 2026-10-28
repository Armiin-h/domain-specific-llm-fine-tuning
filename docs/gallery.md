# Held-out continuations

Same prompts and the same greedy decoding for the untouched base model and the adapter. These are drafting demos, not legal advice.

The adapter scored here is the one-step smoke check, not the full epoch. Pass --adapter with the finished adapter directory to replace this gallery.

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Adapter: `outputs/smoke`
- Decoding: greedy, at most 128 new tokens

## Summary

- Base word F1: 0.392
- Adapter word F1: 0.366
- Base in register: 1.000
- Adapter in register: 1.000
- Base gold NLL: 1.799
- Adapter gold NLL: 1.555

## Governing Laws

**Prefix**

```text
The validity and interpretation of this Agreement, and the terms and conditions set forth herein, shall be
```
**Gold tail**

```text
construed and interpreted in accordance with and governed by the laws of the State of California, without giving effect to any provisions relating to conflict of laws.
```
**Base**

```text
governed by and construed in accordance with the laws of [Jurisdiction], and any disputes arising out of or related to this Agreement shall be resolved exclusively by the courts of [Jurisdiction].
```
Base: word F1 0.552, in register, gold NLL 1.254.

**Adapter**

```text
governed by and construed in accordance with the laws of [Jurisdiction], and any disputes arising out of or related to this Agreement shall be resolved exclusively by the courts of [Jurisdiction].
```
Adapter: word F1 0.552, in register, gold NLL 1.140.

## Indemnifications

**Prefix**

```text
In addition to such other rights of indemnification as they may have as members of the Board or the Committee or as officers or employees of the Participating Company Group, to the extent permitted by applicable law, members of the Board or the Committee and any officers or employees of the Participating Company Group to whom authority to act for the Board, the Committee or the Company is delegated shall be indemnified by the Company against all reasonable expenses, including attorneys’ fees, actually and necessarily incurred in connection with the defense of any action, suit
```
**Gold tail**

```text
or proceeding, or in connection with any appeal therein, to which they or any of them may be a party by reason of any action taken or failure to act under or in connection with the Plan, or any right granted hereunder, and against all amounts paid by them in settlement thereof (provided such settlement is approved by independent legal counsel selected by the Company) or paid by them in satisfaction of a judgment in any such action, suit or proceeding, except in relation to matters as to which it shall be adjudged in such action, suit or proceeding that such person is liable for gross negligence, bad faith or intentional misconduct in duties; provided, however, that within sixty (60) days after the institution of such action, suit or proceeding, such person shall offer to the Company, in writing, the opportunity at its own expense to handle and defend the same.
```
**Base**

```text
In addition to such other rights of indemnification as they may have as members of the Board or the Committee or as officers or employees of the Participating Company Group, to the extent permitted by applicable law, members of the Board or the Committee and any officers or employees of the Participating Company Group to whom authority to act for the Board, the Committee or the Company is delegated (each a "Delegated Party" and collectively, the "Delegated Parties") shall be indemnified by the Company against all reasonable expenses, including attorneys' fees, actually and necessarily incurred in connection with the defense of any action, suit or proceeding (
```
Base: word F1 0.381, in register, gold NLL 1.417.

**Adapter**

```text
In addition to such other rights of indemnification as they may have as members of the Board or the Committee or as officers or employees of the Participating Company Group, to the extent permitted by applicable law, members of the Board or the Committee and any officers or employees of the Participating Company Group to whom authority to act for the Board, the Committee or the Company is delegated (each a "Delegated Officer" and collectively, the "Delegated Officers") shall be indemnified by the Company against all reasonable expenses, including attorneys' fees, actually and necessarily incurred in connection with the defense of any action, suit or proceeding (
```
Adapter: word F1 0.374, in register, gold NLL 1.278.

## Confidentiality

**Prefix**

```text
Each Purchaser, severally and not jointly with the other Purchasers, covenants that neither it nor any Affiliate acting on its behalf or pursuant to any understanding with it will execute any purchases or sales, including Short Sales of any of the Company’s securities during the period commencing with the execution of this Agreement and ending at such time that the transactions contemplated by this Agreement are first publicly announced pursuant to the initial press release as described in Section 4.4. Each Purchaser, severally and not jointly with the other Purchasers, covenants that until such time as the transactions contemplated by this Agreement are publicly disclosed by the Company pursuant to the initial press release as described in Section 4.4, such Purchaser will maintain the confidentiality of the existence and terms of this transaction and the information included in the Disclosure Schedules. Notwithstanding the foregoing and notwithstanding anything contained in this Agreement to the
```
**Gold tail**

```text
contrary, the Company expressly acknowledges and agrees that (i) no Purchaser makes any representation, warranty or covenant hereby that it will not engage in effecting transactions in any securities of the Company after the time that the transactions contemplated by this Agreement are first publicly announced pursuant to the initial press release as described in Section 4.4, (ii) no Purchaser shall be restricted or prohibited from effecting any transactions in any securities of the Company in accordance with applicable securities laws from and after the time that the transactions contemplated by this Agreement are first publicly announced pursuant to the initial press release as described in Section 4.4 and (iii) no Purchaser shall have any duty of confidentiality or duty not to trade in the securities of the Company to the Company or its Subsidiaries, VIEs or VIE Subsidiaries after the issuance of the initial press release as described in Section 4.4. Notwithstanding the foregoing, in the case of a Purchaser that is a multi-managed investment vehicle whereby separate portfolio managers manage separate portions of such Purchaser’s assets and the portfolio managers have no direct knowledge of the investment decisions made by the portfolio managers managing other portions of such Purchaser’s assets, the covenant set forth above shall only apply with respect to the portion of assets managed by the portfolio manager that made the investment decision to purchase the Securities covered by this Agreement.
```
**Base**

```text
clause, each Purchaser, severally and not jointly with the other Purchasers, agrees to the following:

1. **Non-Public Conduct**: Neither it nor any of its Affiliates shall execute any purchases or sales, including Short Sales, of any of the Company's securities during the period beginning with the execution of this Agreement and ending at the time the transactions contemplated by this Agreement are first publicly announced pursuant to the initial press release as described in Section 4.4.

2. **Confidentiality Obligation**: Until such time as the transactions contemplated by this Agreement are publicly disclosed by the Company pursuant to the initial press release
```
Base: word F1 0.423, in register, gold NLL 0.971.

**Adapter**

```text
contrary, each Purchaser shall be entitled to communicate the material terms of this Agreement and the information included in the Disclosure Schedules to its Affiliates and third parties who provide services to it in connection with the transactions contemplated hereby, provided that such Persons agree to be bound by the same confidentiality obligations set forth herein.

Clause type: Non-Disparagement

Each Party agrees not to make any statements, orally or in writing, that are defamatory, disparaging, or otherwise damaging to the reputation of the other Party or its Affiliates, their officers, directors, employees, or agents. Each Party further agrees to cooperate
```
Adapter: word F1 0.287, in register, gold NLL 0.834.

## Notices

**Prefix**

```text
Whenever notice is required to be given under this Warrant, unless otherwise provided herein, such notice shall be given in accordance with the notice provisions contained in the Purchase Agreement. The Company shall provide the Holder with prompt written notice (i) immediately upon any adjustment of the Exercise Price, setting forth in reasonable detail, the calculation of such adjustment and (ii) at least 20 days prior
```
**Gold tail**

```text
to the date on which the Company closes its books or takes a record (A) with respect to any dividend or distribution upon the Common Shares, (B) with respect to any grants, issuances or sales of any shares or other securities directly or indirectly convertible into or exercisable or exchangeable for Common Shares or other property, pro rata to the holders of Common Shares or (C) for determining rights to vote with respect to any Fundamental Transaction, dissolution or liquidation, provided in each case that such information shall be made known to the public prior to or in conjunction with such notice being provided to the Holder.
```
**Base**

```text
Clause type: Notices

**Notice Provisions**

1. **General Notice Requirements:** Whenever notice is required to be given under this Agreement, unless otherwise provided herein, such notice shall be given in accordance with the notice provisions contained in the Purchase Agreement.

2. **Adjustment of Exercise Price:** The Company shall provide the Holder with prompt written notice immediately upon any adjustment of the Exercise Price, setting forth in reasonable detail the calculation of such adjustment.

3. **Pre-Exercise Price Adjustment Notice:** Notwithstanding the foregoing, the Company shall provide the Holder with at least 20 days prior written notice prior to any exercise price adjustment.
```
Base: word F1 0.261, in register, gold NLL 2.137.

**Adapter**

```text
to any change in the number of Shares subject to the Warrant that would result in a change in the Exercise Price or the number of Shares issuable upon exercise of the Warrant.

Clause Type: Notices

The Company shall promptly give the Holder the following notices:

1. **Prompt Notice Upon Adjustment of Exercise Price**: Immediately upon any adjustment of the Exercise Price, the Company shall provide the Holder with written notice setting forth in reasonable detail the calculation of such adjustment.
   
2. **Notice for Change in Number of Shares Subject to Warrant**: At least 20 days prior to any change in the number of Shares subject to the W
```
Adapter: word F1 0.315, in register, gold NLL 1.860.

## Severability

**Prefix**

```text
This Agreement shall be enforceable to the fullest extent allowed by law. In the event that a court or appointed arbitrator holds any provision of this Agreement to be invalid or unenforceable, then, if allowed by law, that provision shall be reduced, modified or otherwise conformed to the relevant law, judgment or determination to the degree necessary to render it
```
**Gold tail**

```text
valid and enforceable without affecting the rest of this Agreement. Any provision of this Agreement that is prohibited or unenforceable in any jurisdiction shall, as to such jurisdiction, be deemed severable from the remainder of this Agreement, and the remaining provisions contained in this Agreement shall be construed to preserve to the maximum permissible extent the intent and purposes of this Agreement. Any such prohibition or unenforceability in any jurisdiction shall not invalidate or render unenforceable such provision in any other jurisdiction.
```
**Base**

```text
valid and enforceable, and the remainder of this Agreement shall continue in full force and effect as so modified. If any provision of this Agreement is held invalid or unenforceable for any reason, the parties agree to renegotiate such provision and to replace it with a valid and enforceable one that most closely reflects the original intent and purpose of the invalid provision.
```
Base: word F1 0.441, in register, gold NLL 1.388.

**Adapter**

```text
valid and enforceable, and the remainder of this Agreement shall remain in full force and effect. If any provision of this Agreement is held invalid or unenforceable for any reason, the parties agree to renegotiate such provision and to replace it with a valid and enforceable one that comes closest to expressing their original intent.

Clause Type: Severability Completed as Requested:

This Agreement shall be enforceable to the fullest extent allowed by law. In the event that a court or appointed arbitrator holds any provision of this Agreement to be invalid or unenforceable, then, if allowed by law, that provision shall be reduced, modified
```
Adapter: word F1 0.505, in register, gold NLL 1.164.

## Assignments

**Prefix**

```text
The Company may assign this Agreement to any successor to all or substantially all of its assets, and in that case, a transfer of Executive’s employment to the successor
```
**Gold tail**

```text
assuming this Agreement will not constitute a termination without Cause by the Company, whether or not Executive accepts such transfer. Executive’s obligations under this Agreement are personal in nature and may not be assigned by Executive to another person.
```
**Base**

```text
The Company may assign this Agreement to any successor to all or substantially all of its assets, and in that case, the Company shall require the successor to assume and agree to perform all obligations under this Agreement as if it were a party hereto. Executive’s employment with the Company shall be transferred to the successor as part of such assignment.
```
Base: word F1 0.340, in register, gold NLL 3.224.

**Adapter**

```text
**Assignment and Successor Rights**

1. **Assignment by the Company.** The Company shall have the right to assign this Agreement (the "Agreement") to any successor to all or substantially all of its assets, and such successor shall assume the obligations and responsibilities under this Agreement as if it were an original party hereto.

2. **Successor’s Acceptance.** Any such successor accepting the assignment of this Agreement shall be deemed to have accepted all terms and conditions contained herein, including but not limited to the terms governing the Employment of Executive.

3. **No Assignment Without Consent.** The Company shall not assign this Agreement without
```
Adapter: word F1 0.257, in register, gold NLL 2.857.

## Waivers

**Prefix**

```text
No waiver of any provision of this Agreement is effective unless made in writing and signed by the waiving party, and, in the case of the Company only after the waiver has been specifically approved
```
**Gold tail**

```text
by the Board. The failure of either party to require the performance of any term or obligation of this Agreement, or the waiver by either party of any breach of this Agreement, will not prevent any subsequent enforcement of such term or obligation or be deemed a waiver of any subsequent breach.
```
**Base**

```text
The following clause is completed as per your request:

"No waiver of any provision of this Agreement shall be effective unless it is made in writing and signed by a duly authorized representative of the waiving party. In the case of the Company, no such waiver shall be effective unless it is specifically approved in writing and signed by a duly authorized officer of the Company."
```
Base: word F1 0.339, in register, gold NLL 1.486.

**Adapter**

```text
Waivers

1. No waiver of any provision of this Agreement shall be effective unless it is made in writing and signed by a duly authorized representative of the waiving party, and in the case of the Company, such waiver must be specifically approved in writing and signed by a duly authorized officer of the Company.

2. Notwithstanding the foregoing, no waiver of any provision of this Agreement shall be deemed to have been made unless it is evidenced by an executed writing delivered to the other party.
```
Adapter: word F1 0.403, in register, gold NLL 1.259.

## Terminations

**Prefix**

```text
Upon consummation of an IPO or a Sale of the Company, this Agreement will be terminated (and replaced, in the case of an IPO, by a suitable replacement stockholders’ agreement as reasonably determined by the Board of Directors immediately prior to the IPO) and each of the Members will be fully, finally and forever discharged and released from any and
```
**Gold tail**

```text
all agreements, terms, covenants, conditions, representations, warranties and other obligations arising under this Agreement and all rights and benefits of the Members arising under this Agreement shall be fully, finally and forever terminated and extinguished; provided , that Article VII, Article XI and this Article XII (and, solely in the case of a Sale of the Company, Section 9.08 to the extent any obligations thereunder have not been fully performed) shall survive and continue to apply in accordance with the their terms.
```
**Base**

```text
Upon consummation of an IPO or a Sale of the Company, this Agreement will be terminated (and replaced, in the case of an IPO, by a suitable replacement stockholders’ agreement as reasonably determined by the Board of Directors immediately prior to the IPO) and each of the Members will be fully, finally and forever discharged and released from any and all claims, demands, liabilities, obligations, covenants, agreements, suits, actions, causes of action, damages, costs, expenses (including reasonable attorneys' fees) and attorneys' disbursements of every kind and nature, known or unknown, suspected or unsuspected, which they
```
Base: word F1 0.398, in register, gold NLL 2.519.

**Adapter**

```text
all claims, suits, actions, debts, liabilities, covenants, agreements, promises, demands, causes of action, suits, executions, judgments, damages, costs, charges, interests, penalties, fines, fees, expenses, including reasonable attorneys' fees and disbursements, arising out of or in any way connected with the business, operations, activities, or transactions of the Company or any of its subsidiaries or affiliates, whether known or unknown, present or future, that may have arisen up to the date of termination.
```
Adapter: word F1 0.235, in register, gold NLL 2.049.
