<script lang="ts">
	const baseUrl = import.meta.env.VITE_BASE_URL
	const bearerToken = import.meta.env.VITE_BEARER_TOKEN
	import { conversation } from '$lib/conversationStore';
	$conversation = [];
	import { streamstore } from '$lib/streamStore';
	$streamstore = [];
	let answer: any = [];
	$: answer = answer;
	let streaming: boolean;
	$: streaming = streaming;
	import { goto } from '$app/navigation';

	export let collection: any;
	export let collections: any;
	export let index: any;
	let prompt: string = '';
	$: prompt = prompt;

	let mode = 0;


	$: mode = mode;


	let searching = false;
	const onKeyPress = async (e) => {
		streaming = true;
		if (e.charCode === 13) {
			let oldprompt = prompt;

			searching = true;
			prompt = '';

			try {
				$conversation = [
					...$conversation,
					{
						json: false,
						bot: false,
						error: false,
						message: oldprompt,
						date: new Date().toLocaleDateString('nb-NO')
					}
				];
				console.log({
					collection: collection,
					prompt: oldprompt,
					temperature: 0
				});
				const response = await fetch(`${baseUrl}/collections/stream`, {
					method: 'POST', // or 'PUT'
					headers: {
						'Content-Type': 'application/json',
						Authorization: `Bearer ${bearerToken}`
					},
					body: JSON.stringify({
						collection: collection,
						prompt: oldprompt,
						temperature: 0
					})
				});

				const stream = response.body;
				const reader = stream.getReader();
				const decoder = new TextDecoder();
				const readStream = async () => {
					while (true) {
						const { done, value } = await reader.read();
						if (done) {
							streaming = false;
							return;
						}

						const decodedValue = decoder.decode(value);
						$streamstore = [...$streamstore, decodedValue];
					}
				};

				await readStream();
				// const fetchResponse = await fetch(`/api/queryGPT`, settings);
				// const data = await fetchResponse.json();

				searching = false;
			} catch (e) {
				$conversation = [
					...$conversation,
					{
						bot: true,
						error: true,
						message: 'Det skjedde en feil, prøv igjen senere...',
						date: new Date().toLocaleDateString('nb-NO')
					}
				];
				searching = false;
				prompt = oldprompt;
				return e;
			}
		}
	};

	$: selected = collection;
	$: streamstoreoutput = '';
	import { fade, fly } from 'svelte/transition';
	const scrollToBottom = (node) => {
		const scroll = () =>
			node.scroll({
				top: node.scrollHeight,
				behavior: 'smooth'
			});
		scroll();

		return { update: scroll };
	};

	$: {
		streamstoreoutput = $streamstore.join('');

		if (streaming === false) {
			let data = $streamstore.join('');
			$conversation = [
				...$conversation,
				{
					json: false,
					bot: true,
					error: false,
					message: data,
					date: new Date().toLocaleDateString('nb-NO')
				}
			];
			$streamstore = [];
		}
	}
</script>

<svelte:window />
<div class="flex flex-col h-screen overflow-x-hidden">
	<div class="px-12 pt-12 pb-12 flex-grow overflow-x-hidden" use:scrollToBottom={$streamstore}>
		<div class="chat chat-start overflow-x-hidden">
			<div class="chat-image avatar">
				<div class="w-16 mask mask-round">
					<img src="/profile-image.png" />
				</div>
			</div>

			<div class="chat-header">
				dagthomasBot
				<time class="text-xs opacity-50"
					>{new Date().toLocaleTimeString('nb-NO', { hour: 'numeric', minute: 'numeric' })}</time
				>
			</div>
			<div class="chat-bubble bg-ll-p-dark">
				<span>Hello there, I am <strong>dagthomasBot</strong>, what can I help you with? </span>
			</div>
		</div>
		{#each $conversation as item}
			{#if item?.bot}
				<div class="chat chat-start" out:fade>
					<div class="chat-image avatar">
						<div class="w-16 mask mask-round">
							<img src="/profile-image.png" />
						</div>
					</div>
					<div class="chat-header">
						dagthomasBot
						<time class="text-xs opacity-50"
							>{item?.data ||
								new Date().toLocaleTimeString('nb-NO', {
									hour: 'numeric',
									minute: 'numeric'
								})}</time
						>
					</div>
					<div class="pre chat-bubble bg-ll-p-dark">
						{item?.message}
					</div>
				</div>
			{:else}
				<div class="chat chat-end" in:fly={{ x: 50, duration: 250 }} out:fade>
					<div class="chat-header">
						User
						<time class="text-xs opacity-50"
							>{item?.data ||
								new Date().toLocaleTimeString('nb-NO', {
									hour: 'numeric',
									minute: 'numeric'
								})}</time
						>
					</div>
					<div class="pre chat-bubble bg-ll-s-1 text-white">
						{item?.message}
					</div>
				</div>
			{/if}
		{/each}
		{#if searching}
			<div class="chat chat-start" in:fly={{ x: -150, duration: 500, delay: 750 }}>
				<div class="chat-image avatar">
					<div class="w-16 mask mask-round">
						<img src="/profile-image.png" />
					</div>
				</div>
				<div class="chat-header" in:fade={{ duration: 250 }}>
					dagthomasBot
					<time class="text-xs opacity-50"
						>{new Date().toLocaleTimeString('nb-NO', {
							hour: 'numeric',
							minute: 'numeric'
						})}</time
					>
				</div>
				{#if streamstoreoutput.length > 0}
					<div class="chat-bubble bg-ll-p-dark" in:fade={{ duration: 750 }}>
						{#each [...streamstoreoutput] as word}
							<span in:fade={{ duration: 75 }} class="pre">{word}</span>
						{/each}
					</div>
				{/if}
			</div>
		{/if}
	</div>
	<div class="navbar mb-4">
		<div class="flex-1 justify-center">
			<input
				type="text"
				placeholder="Ask a question"
				class="input input-bordered w-3/6 border-ll-s-1 border-2"
				bind:value={prompt}
				name="message"
				on:keypress={onKeyPress}
			/>
		</div>
	</div>
	<div class="absolute right-0 top-0 mt-8 mr-8">
		<h1 class="mb-2 ml-1 font-bold">Collections</h1>
		<select
			class="select select-bordered max-w-xs select-sm"
			bind:value={selected}
			on:change={() => goto(selected)}
		>
			{#each collections as collection}
				<option value={collection.name}> {collection.name}</option>
			{/each}
		</select>
	</div>
</div>

<style>
	.pre {
		white-space: pre-line;
	}
	/* .chat-image {
		align-self: flex-start;
	} */
	:global(html) {
		scroll-behavior: smooth;
	}
</style>
